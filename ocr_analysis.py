import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_SEED = 20261017
DEFAULT_RESAMPLES = 10000
CI_LEVEL = 95.0
ELIGIBILITY_TOLERANCE = 1e-6
TIE_DECIMALS = 9
TIME_TIE_FRACTION = 0.05
ENGINE_ORDER = ["EasyOCR", "Tesseract", "PaddleOCR"]
EXPECTED_SETTINGS = 101
DEFAULT_RESULTS_FILE = "ocr_experiment_answersheet_all_results.csv"


def to_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("none", "nan", "na", "n/a", "null"):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        rows = data.get("results") if isinstance(data, dict) else data
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{path} does not look like a results file made by the OCR GUI.")
        return rows

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return [{key: value for key, value in row.items() if key is not None}
                for row in csv.DictReader(file)]


def expand_inputs(paths):
    files = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            found = sorted(p for p in entry.iterdir() if p.suffix.lower() in (".csv", ".json"))
            files.extend(found)
        elif entry.exists():
            files.append(entry)
        else:
            raise FileNotFoundError(f"Input not found: {entry}")
    if not files:
        raise FileNotFoundError("No .csv or .json result files were found.")
    return files


def setting_label(strength, block, c_value):
    return f"strength={strength:g}, block={int(block)}, C={int(c_value)}"


def clean_row(row):
    image_id = str(row.get("image_id", "")).strip()
    engine = str(row.get("engine", "")).strip()
    if not image_id or not engine:
        return None

    if str(row.get("preprocessing", "")).strip() == "raw":
        label, params = "raw", (None, None, None)
    else:
        strength, block, c_value = to_float(row.get("strength")), to_float(row.get("block")), to_float(row.get("C"))
        if strength is None or block is None or c_value is None:
            return None
        label, params = setting_label(strength, block, c_value), (strength, int(block), int(c_value))

    record = {
        "image_id": image_id, "engine": engine, "setting": label, "params": params,
        "name_cer": to_float(row.get("name_cer")), "lrn_cer": to_float(row.get("lrn_cer")),
        "pooled_cer": to_float(row.get("pooled_cer")),
        "processing_seconds": to_float(row.get("processing_seconds")),
        "preprocess_seconds": to_float(row.get("preprocess_seconds")),
        "ocr_seconds": to_float(row.get("ocr_seconds")),
        "score_seconds": to_float(row.get("stage_score_seconds")),
        "status": str(row.get("status", "")).strip(),
        "name_model": str(row.get("name_model", "") or "").strip(),
        "gt_name": row.get("gt_name_norm", ""), "gt_lrn": row.get("gt_lrn_norm", ""),
        "pred_name": row.get("pred_name_norm", ""), "pred_lrn": row.get("pred_lrn_norm", ""),
    }
    record["valid"] = (
        record["status"] != "invalid_ground_truth"
        and None not in (record["name_cer"], record["lrn_cer"], record["pooled_cer"])
    )
    return record


def merge_files(files):
    by_image = {}
    source = {}
    notes = []
    for path in files:
        per_image = {}
        for raw_row in read_rows(path):
            record = clean_row(raw_row)
            if record is not None:
                per_image.setdefault(record["image_id"], {})[(record["engine"], record["setting"])] = record
        for image_id, rows in per_image.items():
            if image_id in by_image:
                notes.append(f"Sheet '{image_id}' from {source[image_id]} was replaced by the one in {path.name}.")
            by_image[image_id] = rows
            source[image_id] = path.name
    records = [record for rows in by_image.values() for record in rows.values()]
    return records, notes


def describe(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def tie_key(stats):
    return (round(stats["pooled"]["mean"], TIE_DECIMALS), round(stats["pooled"]["sd"], TIE_DECIMALS))


def order_tied_group(group):
    remaining = list(group)
    ordered = []
    while remaining:
        times = [c["time_mean"] if c["time_mean"] is not None else math.inf for c in remaining]
        fastest = min(times)
        limit = fastest * (1 + TIME_TIE_FRACTION) if math.isfinite(fastest) else math.inf
        near = [c for c, t in zip(remaining, times) if t <= limit]
        pick = min(near, key=lambda c: c["params"])
        ordered.append(pick)
        remaining = [c for c in remaining if c is not pick]
    return ordered


def percentile_interval(values):
    low, high = np.percentile(values, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
    return float(low), float(high)


def build_dataset(records, expected_sheets, allow_mixed):
    warnings = []

    engines = [e for e in ENGINE_ORDER if any(r["engine"] == e for r in records)]
    engines += sorted({r["engine"] for r in records} - set(engines))
    settings = sorted({r["setting"] for r in records}, key=lambda s: (s != "raw", s))
    if "raw" not in settings:
        raise ValueError("There is no raw baseline in these results, so nothing can be compared against raw.")
    if len(settings) != EXPECTED_SETTINGS:
        warnings.append(f"Found {len(settings)} settings (expected {EXPECTED_SETTINGS}: 100 combos + raw).")

    name_models = {}
    for engine in engines:
        models = sorted({r["name_model"] for r in records if r["engine"] == engine and r["name_model"]})
        name_models[engine] = models
        if len(models) > 1:
            message = f"{engine}: the Name field was read by different models: {', '.join(models)}."
            if not allow_mixed:
                raise ValueError(message + " Use one model throughout, or pass --allow-mixed-models.")
            warnings.append(message)

    lookup = {}
    for r in records:
        lookup[(r["image_id"], r["engine"], r["setting"])] = r
    image_ids = sorted({r["image_id"] for r in records})

    complete, excluded = [], []
    for image_id in image_ids:
        missing = [(e, s) for e in engines for s in settings
                   if (image_id, e, s) not in lookup or not lookup[(image_id, e, s)]["valid"]]
        if missing:
            excluded.append((image_id, len(missing), len(engines) * len(settings)))
        else:
            complete.append(image_id)

    if len(complete) < 2:
        raise ValueError(f"Only {len(complete)} sheet(s) have a complete set of results; at least 2 are needed.")
    if expected_sheets and len(complete) != expected_sheets:
        warnings.append(f"Expected {expected_sheets} sheets but {len(complete)} are complete.")
    for image_id, missing, total in excluded:
        warnings.append(f"Sheet '{image_id}' left out: {missing} of {total} engine/setting results are missing or invalid.")

    return {
        "engines": engines, "settings": settings, "sheets": complete, "lookup": lookup,
        "name_models": name_models,
    }, warnings


def processing_time(dataset, image_id, engine, setting):
    record = dataset["lookup"][(image_id, engine, setting)]
    if record["processing_seconds"] is not None:
        return record["processing_seconds"]
    shared = [dataset["lookup"][(image_id, e, setting)]["preprocess_seconds"] for e in dataset["engines"]]
    shared = [value for value in shared if value is not None]
    if record["ocr_seconds"] is None or not shared:
        return None
    return max(shared) + record["ocr_seconds"] + (record["score_seconds"] or 0.0)


def engine_statistics(dataset, engine):
    sheets = dataset["sheets"]
    result = {}
    for setting in dataset["settings"]:
        rows = [dataset["lookup"][(sheet, engine, setting)] for sheet in sheets]
        vectors = {
            "name": np.array([r["name_cer"] for r in rows]),
            "lrn": np.array([r["lrn_cer"] for r in rows]),
            "pooled": np.array([r["pooled_cer"] for r in rows]),
        }
        times = [processing_time(dataset, sheet, engine, setting) for sheet in sheets]
        times = [t for t in times if t is not None]
        result[setting] = {
            "setting": setting, "params": None, "vectors": vectors,
            "name": describe(vectors["name"]), "lrn": describe(vectors["lrn"]),
            "pooled": describe(vectors["pooled"]),
            "time_mean": float(np.mean(times)) if times else None,
        }
        result[setting]["params"] = rows[0]["params"] if rows[0]["params"][0] is not None else (-1, -1, -1)
    return result


def select_for_engine(stats):
    raw = stats["raw"]
    combos = [s for key, s in stats.items() if key != "raw"]

    def not_worse(combo):
        return all(combo[field]["mean"] <= raw[field]["mean"] + ELIGIBILITY_TOLERANCE
                   for field in ("name", "lrn", "pooled"))

    eligible = [c for c in combos if not_worse(c)]
    eligible_ids = {id(c) for c in eligible}
    for combo in combos:
        combo["eligible"] = id(combo) in eligible_ids

    groups = {}
    for combo in eligible:
        groups.setdefault(tie_key(combo), []).append(combo)
    ordered = []
    for key in sorted(groups):
        group = groups[key]
        ordered.extend(order_tied_group(group) if len(group) > 1 else group)
    for position, combo in enumerate(ordered, start=1):
        combo["order"] = position
    for combo in combos:
        combo.setdefault("order", None)

    first_key = tie_key(ordered[0]) if ordered else None
    co_best = [c for c in ordered if tie_key(c) == first_key] if ordered else []
    return {
        "raw": raw, "combos": combos, "eligible": eligible, "ordered": ordered,
        "top": ordered[0] if ordered else None,
        "second": ordered[1] if len(ordered) > 1 else None,
        "co_best_exact": co_best,
    }


def bootstrap_engine(engine, selection, index_matrix):
    rows = []
    raw, top, second = selection["raw"], selection["top"], selection["second"]

    def resampled_mean(vector):
        return vector[index_matrix].mean(axis=1)

    def add_interval_rows(label, combo):
        for field in ("pooled", "name", "lrn"):
            low, high = percentile_interval(resampled_mean(combo["vectors"][field]))
            rows.append({"engine": engine, "comparison": f"mean {field} CER of {label}",
                         "setting_a": combo["setting"], "setting_b": "", "metric": field,
                         "estimate": combo[field]["mean"], "ci_low": low, "ci_high": high,
                         "conclusion": ""})

    add_interval_rows("raw", raw)
    if top is None:
        return rows
    add_interval_rows("top combo", top)

    for field in ("pooled", "name", "lrn"):
        difference = raw["vectors"][field] - top["vectors"][field]
        low, high = percentile_interval(resampled_mean(difference))
        if field == "pooled":
            conclusion = ("Preprocessing improved on raw (interval entirely above 0)." if low > 0
                          else "No evidence of improvement over raw (interval includes 0).")
        else:
            conclusion = "interval entirely above 0" if low > 0 else "interval includes 0"
        rows.append({"engine": engine, "comparison": f"raw - top combo ({field} CER)",
                     "setting_a": "raw", "setting_b": top["setting"], "metric": field,
                     "estimate": float(difference.mean()), "ci_low": low, "ci_high": high,
                     "conclusion": conclusion})

    if second is not None:
        difference = second["vectors"]["pooled"] - top["vectors"]["pooled"]
        low, high = percentile_interval(resampled_mean(difference))
        exact_tie = tie_key(second) == tie_key(top)
        if exact_tie:
            conclusion = "Exact tie: co-best."
        elif low > 0 or high < 0:
            conclusion = "Interval excludes 0: the top combo is the single best."
        else:
            conclusion = "Interval includes 0: the top two are not distinguishable (co-best)."
        rows.append({"engine": engine, "comparison": "#2 - #1 (pooled CER)",
                     "setting_a": second["setting"], "setting_b": top["setting"], "metric": "pooled",
                     "estimate": float(difference.mean()), "ci_low": low, "ci_high": high,
                     "conclusion": conclusion})
    return rows


def parameter_effects(engine, selection):
    rows = []
    combos = selection["combos"]
    names = ("strength", "block", "C")
    for index, name in enumerate(names):
        for value in sorted({c["params"][index] for c in combos}):
            members = [c for c in combos if c["params"][index] == value]
            rows.append({
                "engine": engine, "table": "marginal (all combos with this value)",
                "parameter": name, "value": value, "combos": len(members),
                "mean_pooled_cer": float(np.mean([c["pooled"]["mean"] for c in members])),
                "mean_name_cer": float(np.mean([c["name"]["mean"] for c in members])),
                "mean_lrn_cer": float(np.mean([c["lrn"]["mean"] for c in members])),
                "eligible_combos": sum(1 for c in members if c["eligible"]),
            })
    top = selection["top"]
    if top is not None:
        for index, name in enumerate(names):
            others = [i for i in range(3) if i != index]
            same = [c for c in combos if all(c["params"][i] == top["params"][i] for i in others)]
            for combo in sorted(same, key=lambda c: c["params"][index]):
                rows.append({
                    "engine": engine,
                    "table": "one-at-a-time from the top combo (other two parameters fixed)",
                    "parameter": name, "value": combo["params"][index], "combos": 1,
                    "mean_pooled_cer": combo["pooled"]["mean"], "mean_name_cer": combo["name"]["mean"],
                    "mean_lrn_cer": combo["lrn"]["mean"], "eligible_combos": int(combo["eligible"]),
                })
    return rows


def top_combo_errors(dataset, engine, selection):
    top = selection["top"]
    if top is None:
        return []
    rows = []
    for sheet in dataset["sheets"]:
        record = dataset["lookup"][(sheet, engine, top["setting"])]
        if record["pooled_cer"] > 0:
            rows.append({
                "engine": engine, "image_id": sheet, "setting": top["setting"],
                "name_cer": record["name_cer"], "lrn_cer": record["lrn_cer"], "pooled_cer": record["pooled_cer"],
                "raw_pooled_cer": dataset["lookup"][(sheet, engine, "raw")]["pooled_cer"],
                "gt_name": record["gt_name"], "pred_name": record["pred_name"],
                "gt_lrn": record["gt_lrn"], "pred_lrn": record["pred_lrn"],
            })
    return rows


def by_condition(dataset, engine, selection, conditions):
    top = selection["top"]
    chosen = top["setting"] if top is not None else "raw"
    groups = {}
    for sheet in dataset["sheets"]:
        groups.setdefault(conditions.get(sheet, "(no condition given)"), []).append(sheet)
    rows = []
    for condition in sorted(groups):
        sheets = groups[condition]
        raw_values = [dataset["lookup"][(s, engine, "raw")]["pooled_cer"] for s in sheets]
        top_values = [dataset["lookup"][(s, engine, chosen)]["pooled_cer"] for s in sheets]
        rows.append({
            "engine": engine, "condition": condition, "sheets": len(sheets), "top_setting": chosen,
            "raw_mean_pooled_cer": float(np.mean(raw_values)), "top_mean_pooled_cer": float(np.mean(top_values)),
            "raw_minus_top": float(np.mean(raw_values) - np.mean(top_values)),
        })
    return rows


def summary_row(engine, dataset, selection, boot_rows):
    top = selection["top"]
    representative = top if top is not None else selection["raw"]
    row = {
        "engine": engine,
        "setting": representative["setting"] if top is not None else "raw (no combo qualified)",
        "strength": top["params"][0] if top is not None else "",
        "block": top["params"][1] if top is not None else "",
        "C": top["params"][2] if top is not None else "",
        "sheets": len(dataset["sheets"]),
        "eligible_combos": len(selection["eligible"]),
        "exactly_tied_combos": len(selection["co_best_exact"]),
        "name_model": "; ".join(dataset["name_models"].get(engine, [])),
    }
    for field in ("name", "lrn", "pooled"):
        for stat in ("mean", "median", "sd", "min", "max"):
            row[f"{field}_{stat}"] = representative[field][stat]
    row["mean_processing_seconds"] = representative["time_mean"] if representative["time_mean"] is not None else ""
    row["raw_pooled_mean"] = selection["raw"]["pooled"]["mean"]

    mine = [b for b in boot_rows if b["engine"] == engine]
    own = next((b for b in mine if b["comparison"] == "mean pooled CER of top combo"), None)
    row["pooled_ci_low"], row["pooled_ci_high"] = (own["ci_low"], own["ci_high"]) if own else ("", "")
    versus_raw = next((b for b in mine if b["comparison"] == "raw - top combo (pooled CER)"), None)
    if versus_raw:
        row["raw_minus_top_pooled"] = versus_raw["estimate"]
        row["raw_minus_top_ci_low"], row["raw_minus_top_ci_high"] = versus_raw["ci_low"], versus_raw["ci_high"]
        row["improved_on_raw"] = "yes" if versus_raw["ci_low"] > 0 else "no"
    else:
        row["raw_minus_top_pooled"] = row["raw_minus_top_ci_low"] = row["raw_minus_top_ci_high"] = ""
        row["improved_on_raw"] = "no (raw recommended)"
    second = next((b for b in mine if b["comparison"] == "#2 - #1 (pooled CER)"), None)
    if top is None:
        row["top_vs_second"] = "n/a"
    elif second is None:
        row["top_vs_second"] = "no second eligible combo"
    elif "single best" in second["conclusion"]:
        row["top_vs_second"] = "single best"
    else:
        row["top_vs_second"] = "co-best"
    return row


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        return False
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def fmt(value, digits=4):
    if value is None or value == "":
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers, table_rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(path, meta, dataset, selections, summary_rows, boot_rows, effect_rows, warnings, notes):
    out = []
    out.append("# OCR experiment: selection and bootstrap analysis\n")
    out.append(f"Generated {meta['generated']} by ocr_analysis.py. Seed: **{meta['seed']}**, "
               f"resamples: **{meta['resamples']}**, interval: {CI_LEVEL:g}% percentile.\n")

    out.append("## Data\n")
    out.append(f"- Sheets used: **{len(dataset['sheets'])}** ({', '.join(dataset['sheets'])})")
    out.append(f"- Settings: {len(dataset['settings'])} (raw plus {len(dataset['settings']) - 1} combos); "
               f"engines: {', '.join(dataset['engines'])}")
    for engine in dataset["engines"]:
        out.append(f"- Name reader for {engine}: {'; '.join(dataset['name_models'][engine]) or 'not recorded'}")
    out.append("- Input files (SHA-256):")
    for item in meta["files"]:
        out.append(f"  - {item['name']} `{item['sha256']}`")
    if notes or warnings:
        out.append("\n### Warnings and notes\n")
        out.extend(f"- {text}" for text in notes + warnings)

    out.append("\n## Result: top combo per engine\n")
    headers = ["Engine", "Top combo", "Pooled CER mean [95% CI]", "Median", "SD", "Min", "Max",
               "Name CER", "LRN CER", "Time/sheet (s)", "Eligible", "Improved on raw", "#1 vs #2"]
    table = []
    for r in summary_rows:
        table.append([r["engine"], r["setting"],
                      f"{fmt(r['pooled_mean'])} [{fmt(r['pooled_ci_low'])}, {fmt(r['pooled_ci_high'])}]",
                      fmt(r["pooled_median"]), fmt(r["pooled_sd"]), fmt(r["pooled_min"]), fmt(r["pooled_max"]),
                      fmt(r["name_mean"]), fmt(r["lrn_mean"]), fmt(r["mean_processing_seconds"], 2),
                      r["eligible_combos"], r["improved_on_raw"], r["top_vs_second"]])
    out.append(markdown_table(headers, table))

    for r in summary_rows:
        engine = r["engine"]
        selection = selections[engine]
        out.append(f"\n### {engine}\n")
        raw = selection["raw"]
        out.append(f"- Raw baseline: pooled {fmt(raw['pooled']['mean'])}, name {fmt(raw['name']['mean'])}, "
                   f"LRN {fmt(raw['lrn']['mean'])}.")
        out.append(f"- Eligible combos: {len(selection['eligible'])} of {len(selection['combos'])}.")
        if selection["top"] is None:
            out.append("- **No combo is at least as good as raw on name, LRN and pooled CER. "
                       "Raw is the recommendation for this engine.**")
            continue
        out.append(f"- Representative top combo: **{selection['top']['setting']}**.")
        if len(selection["co_best_exact"]) > 1:
            names = [c["setting"] for c in selection["co_best_exact"]]
            shown = ", ".join(names[:15]) + (f", ... ({len(names)} in total)" if len(names) > 15 else "")
            out.append(f"- {len(names)} combos are exactly tied (co-best): {shown}. "
                       "The representative is the one with the lowest mean processing time "
                       "(within 5% counts as equal, then lower strength, block, C).")
        for b in boot_rows:
            if b["engine"] == engine and (b["comparison"].startswith("raw - top") or b["comparison"].startswith("#2")):
                out.append(f"- {b['comparison']}: {fmt(b['estimate'])} [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}]. "
                           f"{b['conclusion']}")

    out.append("\n## Mean CER per parameter value\n")
    for engine in dataset["engines"]:
        marginal = [e for e in effect_rows if e["engine"] == engine and e["table"].startswith("marginal")]
        if not marginal:
            continue
        out.append(f"**{engine}** (mean over all combos with that value)\n")
        out.append(markdown_table(["Parameter", "Value", "Combos", "Pooled CER", "Name CER", "LRN CER", "Eligible"],
                                  [[e["parameter"], e["value"], e["combos"], fmt(e["mean_pooled_cer"]),
                                    fmt(e["mean_name_cer"]), fmt(e["mean_lrn_cer"]), e["eligible_combos"]]
                                   for e in marginal]))
        out.append("")

    out.append("## Rules applied\n")
    out.append("Eligibility (strict, mean name, LRN and pooled CER each <= raw, tolerance 1e-6); "
               "order by mean pooled CER, then lower SD; exact ties are co-best and the representative has the "
               "lowest mean processing time (within 5% equal, then lower strength, block, C). "
               "Bootstrap resamples whole sheets, fixed seed, same resamples for every comparison; "
               "it describes the result and never changes the selection.")
    out.append("\n## Limits to state in the paper\n")
    out.append("- The top combo was chosen as the best of many on these same sheets, so its scores are in-sample "
               "and likely slightly optimistic. There was no held-out check.")
    out.append("- 'Not distinguishable' is not 'equivalent': with this many sheets the test has limited power.")
    out.append("")
    meta_text = (f"Python {meta['python']}, numpy {meta['numpy']}, {meta['platform']}.")
    out.append(meta_text)
    Path(path).write_text("\n".join(out), encoding="utf-8")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_conditions(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return {str(row["image_id"]).strip(): str(row["condition"]).strip() for row in csv.DictReader(file)}


def run_analysis(inputs, output_dir, seed=DEFAULT_SEED, resamples=DEFAULT_RESAMPLES,
                 conditions_path=None, expected_sheets=None, allow_mixed=False):
    files = expand_inputs(inputs)
    records, notes = merge_files(files)
    dataset, warnings = build_dataset(records, expected_sheets, allow_mixed)

    n = len(dataset["sheets"])
    rng = np.random.default_rng(seed)
    index_matrix = rng.integers(0, n, size=(resamples, n))

    selections, boot_rows, effect_rows, error_rows, summary_rows, all_rows, condition_rows = {}, [], [], [], [], [], []
    conditions = read_conditions(conditions_path) if conditions_path else None

    for engine in dataset["engines"]:
        stats = engine_statistics(dataset, engine)
        selection = select_for_engine(stats)
        selections[engine] = selection
        engine_boot = bootstrap_engine(engine, selection, index_matrix)
        boot_rows.extend(engine_boot)
        effect_rows.extend(parameter_effects(engine, selection))
        error_rows.extend(top_combo_errors(dataset, engine, selection))
        summary_rows.append(summary_row(engine, dataset, selection, engine_boot))
        if conditions is not None:
            condition_rows.extend(by_condition(dataset, engine, selection, conditions))
        for combo in [selection["raw"]] + sorted(selection["combos"], key=lambda c: (c["order"] is None, c["order"] or 0, c["setting"])):
            row = {"engine": engine, "setting": combo["setting"],
                   "strength": combo["params"][0] if combo["setting"] != "raw" else "",
                   "block": combo["params"][1] if combo["setting"] != "raw" else "",
                   "C": combo["params"][2] if combo["setting"] != "raw" else "",
                   "is_raw": combo["setting"] == "raw",
                   "eligible": "" if combo["setting"] == "raw" else combo["eligible"],
                   "order": "" if combo["setting"] == "raw" or combo.get("order") is None else combo["order"],
                   "sheets": n}
            for field in ("name", "lrn", "pooled"):
                for stat in ("mean", "median", "sd", "min", "max"):
                    row[f"{field}_{stat}"] = combo[field][stat]
            row["mean_processing_seconds"] = combo["time_mean"] if combo["time_mean"] is not None else ""
            all_rows.append(row)

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"), "seed": seed, "resamples": resamples,
        "ci_level": CI_LEVEL, "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(),
        "files": [{"name": str(p), "sha256": sha256_of(p)} for p in files],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary_table.csv", summary_rows)
    write_csv(output_dir / "all_combos.csv", all_rows)
    write_csv(output_dir / "bootstrap.csv", boot_rows)
    write_csv(output_dir / "parameter_effects.csv", effect_rows)
    write_csv(output_dir / "top_combo_errors.csv", error_rows,
              ["engine", "image_id", "setting", "name_cer", "lrn_cer", "pooled_cer", "raw_pooled_cer",
               "gt_name", "pred_name", "gt_lrn", "pred_lrn"])
    if condition_rows:
        write_csv(output_dir / "by_condition.csv", condition_rows)
    write_report(output_dir / "analysis_report.md", meta, dataset, selections, summary_rows,
                 boot_rows, effect_rows, warnings, notes)
    payload = {"meta": meta, "warnings": notes + warnings, "sheets_used": dataset["sheets"],
               "summary": summary_rows, "bootstrap": boot_rows, "parameter_effects": effect_rows,
               "all_combos": all_rows, "top_combo_errors": error_rows, "by_condition": condition_rows}
    (output_dir / "analysis_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Selection and bootstrap analysis of the OCR experiment results.")
    parser.add_argument("inputs", nargs="*", help="result files (.csv/.json) or folders; later files replace earlier ones per image")
    parser.add_argument("-o", "--output", default="ocr_analysis_output", help="output folder (default: ocr_analysis_output)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"bootstrap seed (default {DEFAULT_SEED}); it is saved in the outputs")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, help=f"bootstrap resamples (default {DEFAULT_RESAMPLES})")
    parser.add_argument("--conditions", help="optional CSV with columns image_id,condition for a per-condition table")
    parser.add_argument("--expected-sheets", type=int, help="warn if the number of complete sheets differs (for example 27)")
    parser.add_argument("--allow-mixed-models", action="store_true", help="do not stop if the results mix name models")
    args = parser.parse_args(argv)

    inputs = args.inputs or ([DEFAULT_RESULTS_FILE] if Path(DEFAULT_RESULTS_FILE).exists() else [])
    if not inputs:
        parser.error("give at least one results file or folder")

    try:
        payload = run_analysis(inputs, args.output, args.seed, args.resamples, args.conditions,
                               args.expected_sheets, args.allow_mixed_models)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Sheets used: {len(payload['sheets_used'])}   seed: {args.seed}   resamples: {args.resamples}")
    for text in payload["warnings"]:
        print(f"  note: {text}")
    print()
    for row in payload["summary"]:
        print(f"{row['engine']:<10} top: {row['setting']:<32} pooled CER {fmt(row['pooled_mean'])} "
              f"[{fmt(row['pooled_ci_low'])}, {fmt(row['pooled_ci_high'])}]  "
              f"improved on raw: {row['improved_on_raw']}  #1 vs #2: {row['top_vs_second']}")
    print(f"\nFiles written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
