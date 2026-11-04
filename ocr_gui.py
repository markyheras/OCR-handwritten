import csv
import json
import statistics
import threading
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import easyocr


CHECKPOINT_FILE = Path("ocr_experiment_answersheet_checkpoint.json")
CUMULATIVE_RESULTS_CSV = Path("ocr_experiment_answersheet_all_results.csv")
AUTOSAVE_EVERY_RESULT = True

TEMPLATE_IMAGE = Path(__file__).resolve().parent / "Answer_Sheet_Name_LRN_A4.png"

SAVE_ALIGNMENT_DEBUG = True
ALIGNMENT_DEBUG_DIR = Path("alignment_debug")

ENGINE_NAMES = ("EasyOCR", "Tesseract", "PaddleOCR")


def normalize(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = unicodedata.normalize("NFC", text.lower())
    return " ".join(text.split())


def restrict_to_enye(text):
    text = unicodedata.normalize("NFC", str(text))
    kept = []
    for char in text:
        if char in "\u00f1\u00d1":
            kept.append(char)
            continue
        plain = "".join(
            part for part in unicodedata.normalize("NFD", char)
            if unicodedata.category(part) != "Mn"
        )
        kept.append(plain)
    return "".join(kept)


def normalize_lrn(text):
    return "".join(normalize(text).split())


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if not s2:
        return len(s1)

    previous = list(range(len(s2) + 1))

    for i, char1 in enumerate(s1):
        current = [i + 1]

        for j, char2 in enumerate(s2):
            insertion = current[j] + 1
            deletion = previous[j + 1] + 1
            substitution = previous[j] + (char1 != char2)
            current.append(min(insertion, deletion, substitution))

        previous = current

    return previous[-1]


def edit_distance(gt, pred):
    return levenshtein(normalize(gt), normalize(pred))


def field_cer(gt, pred):
    gt = normalize(gt)
    pred = normalize(pred)

    if not gt:
        return None

    return edit_distance(gt, pred) / len(gt)


def safe_mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def format_duration(seconds):
    if seconds is None:
        return "N/A"

    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"

    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


RESULT_FILE_TYPES = (".csv", ".json")


def read_results_file(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            rows = data.get("results") if isinstance(data, dict) else data
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError("it does not look like a results file made by this tool")
            return rows

        with open(path, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            return [{key: value for key, value in row.items() if key is not None} for row in reader]

    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
        raise ValueError(
            f"The existing file could not be read, so it was left untouched.\n\n{error}"
        ) from error


def merge_result_rows(existing_rows, new_rows):
    new_ids = {row.get("image_id", "") for row in new_rows}
    existing_ids = {row.get("image_id", "") for row in existing_rows}

    kept_rows = [row for row in existing_rows if row.get("image_id", "") not in new_ids]
    merged = kept_rows + list(new_rows)

    stats = {
        "added": len(new_ids - existing_ids),
        "replaced": len(new_ids & existing_ids),
        "kept": len({row.get("image_id", "") for row in kept_rows}),
        "total_rows": len(merged),
    }
    return merged, stats


def write_result_rows(path, rows, csv_fieldnames):
    path = Path(path)
    temporary_file = path.with_name(path.name + ".tmp")

    if path.suffix.lower() == ".json":
        payload = {"saved_at": datetime.now().isoformat(timespec="seconds"), "results": rows}
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
    else:
        fieldnames = list(csv_fieldnames)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(temporary_file, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)

    temporary_file.replace(path)


def save_results_merged(path, new_rows, csv_fieldnames):
    path = Path(path)
    if path.suffix.lower() not in RESULT_FILE_TYPES:
        raise ValueError("Choose a file name ending in .csv or .json.")

    existing_rows = read_results_file(path)
    merged, stats = merge_result_rows(existing_rows, new_rows)
    write_result_rows(path, merged, csv_fieldnames)
    return stats


def pooled_cer(gt_fields, pred_fields):
    total_distance = 0
    total_characters = 0

    for gt, pred in zip(gt_fields, pred_fields):
        gt = normalize(gt)
        pred = normalize(pred)

        if not gt:
            continue

        total_distance += edit_distance(gt, pred)
        total_characters += len(gt)

    if total_characters == 0:
        return None

    return total_distance / total_characters


def improvement(raw_value, current_value):
    if raw_value is None or current_value is None:
        return None
    return raw_value - current_value


def improvement_percent(raw_value, current_value):
    if raw_value is None or current_value is None or raw_value == 0:
        return None
    return ((raw_value - current_value) / raw_value) * 100.0


def _add_time(timings, stage, start):
    if timings is not None:
        timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - start)


def pipeline_raw(image, timings=None):
    start = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _add_time(timings, "grayscale", start)
    return gray


def denoise_gray(gray, h=10, template_window=7, search_window=21):
    return cv2.fastNlMeansDenoising(
        gray, h=h, templateWindowSize=template_window, searchWindowSize=search_window
    )


def mild_fractional(gray, strength=0.30):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    enhanced = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    result = (1.0 - strength) * gray.astype(np.float32) + strength * enhanced
    return np.clip(result, 0, 255).astype(np.uint8)


def pipeline_frac_adaptive(image, strength=0.30, block=15, c_value=5, timings=None):
    gray = pipeline_raw(image, timings)

    start = time.perf_counter()
    denoised = denoise_gray(gray)
    _add_time(timings, "denoise", start)

    start = time.perf_counter()
    fractional = mild_fractional(denoised, strength=strength)
    _add_time(timings, "sharpen", start)

    block = int(block)
    if block <= 1 or block % 2 == 0:
        raise ValueError("Block size must be odd and greater than 1.")

    start = time.perf_counter()
    result = cv2.adaptiveThreshold(
        fractional, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c_value
    )
    _add_time(timings, "threshold", start)
    return result


STRENGTHS = [0.20, 0.30, 0.40, 0.50]
BLOCKS = [11, 15, 21, 25, 31]
C_VALUES = [3, 5, 8, 10, 12]

COMBINATIONS = [
    (strength, block, c_value)
    for strength in STRENGTHS
    for block in BLOCKS
    for c_value in C_VALUES
]


TEMPLATE_W, TEMPLATE_H = 2480, 3508

TEMPLATE_MARKS = np.array(
    [[244.5, 372.5], [2236.5, 376.5], [2244.5, 3214.5], [238.5, 3214.5]],
    dtype=np.float32,
)

FIELD_BOXES = {
    "name": (543, 518, 1114, 97),
}

LRN_X_EDGES = [540, 611, 676, 741, 806, 870, 935, 1000,
               1065, 1129, 1194, 1259, 1330]
LRN_Y0, LRN_Y1 = 677, 765

CROP_INSET = 10
LRN_CELL_INSET_X = 8
LRN_CELL_INSET_Y = 6
ORIENT_SCALE = 8

_TEMPLATE_SMALL = None


def order_clockwise(points):
    pts = np.asarray(points, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def find_marks(image_bgr):
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr
    )

    scale = 1600.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)

    kernel_size = int(0.07 * max(small.shape)) | 1
    background = cv2.morphologyEx(
        small, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    normalized = small.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    binary = (normalized < 0.45).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = small.shape[0] * small.shape[1]

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (image_area * 0.00015 < area < image_area * 0.02):
            continue

        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        if min(rect_w, rect_h) == 0:
            continue

        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        fill = area / (rect_w * rect_h)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0

        if aspect < 1.5 and fill > 0.80 and solidity > 0.90:
            moments = cv2.moments(contour)
            candidates.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))

    if len(candidates) < 4:
        raise RuntimeError(
            f"only {len(candidates)} of 4 corner marks found "
            "(all four black squares must be visible and in focus)"
        )

    points = np.array(candidates, dtype=np.float32)
    image_corners = np.array(
        [[0, 0], [small.shape[1], 0], [small.shape[1], small.shape[0]], [0, small.shape[0]]],
        dtype=np.float32,
    )

    chosen = [
        points[np.argmin(np.linalg.norm(points - corner, axis=1))]
        for corner in image_corners
    ]

    if len({tuple(np.round(point)) for point in chosen}) < 4:
        raise RuntimeError("corner marks were ambiguous; retake the photo")

    chosen = np.array(chosen, dtype=np.float32) / scale

    quad_area = cv2.contourArea(chosen.reshape(-1, 1, 2))
    if quad_area < 0.05 * gray.shape[0] * gray.shape[1]:
        raise RuntimeError("the sheet is too small in the picture")

    return chosen


def _template_small():
    global _TEMPLATE_SMALL

    if _TEMPLATE_SMALL is None and TEMPLATE_IMAGE.exists():
        template = cv2.imread(str(TEMPLATE_IMAGE), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            template = cv2.resize(
                template,
                (TEMPLATE_W // ORIENT_SCALE, TEMPLATE_H // ORIENT_SCALE),
                interpolation=cv2.INTER_AREA,
            )
            _TEMPLATE_SMALL = cv2.GaussianBlur(template, (5, 5), 0)

    return _TEMPLATE_SMALL


def align_to_template(image_bgr):
    marks = order_clockwise(find_marks(image_bgr))
    template_small = _template_small()

    if template_small is None:
        start = int(np.argmin(marks.sum(axis=1)))
        candidates = [np.roll(marks, -start, axis=0)]
    else:
        candidates = [np.roll(marks, -shift, axis=0) for shift in range(4)]

    best_matrix = None
    best_score = -2.0
    shrink = None
    size = None

    if template_small is not None:
        gray = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if image_bgr.ndim == 3
            else image_bgr
        )
        gray = cv2.GaussianBlur(gray, (0, 0), ORIENT_SCALE / 2.0)
        shrink = np.diag([1.0 / ORIENT_SCALE, 1.0 / ORIENT_SCALE, 1.0])
        size = (template_small.shape[1], template_small.shape[0])

    for points in candidates:
        matrix = cv2.getPerspectiveTransform(points, TEMPLATE_MARKS)

        if template_small is None:
            best_matrix = matrix
            break

        warped = cv2.warpPerspective(gray, shrink @ matrix, size, borderValue=255)
        score = float(cv2.matchTemplate(warped, template_small, cv2.TM_CCOEFF_NORMED)[0, 0])

        if score > best_score:
            best_score = score
            best_matrix = matrix

    return cv2.warpPerspective(
        image_bgr, best_matrix, (TEMPLATE_W, TEMPLATE_H),
        flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255),
    )


def extract_field_crops(image_bgr, timings=None):
    start = time.perf_counter()
    aligned = align_to_template(image_bgr)
    _add_time(timings, "align", start)

    start = time.perf_counter()
    crops = {}
    for field, (x, y, w, h) in FIELD_BOXES.items():
        crops[field] = aligned[
            y + CROP_INSET : y + h - CROP_INSET,
            x + CROP_INSET : x + w - CROP_INSET,
        ].copy()

    crops["lrn_row"] = aligned[LRN_Y0:LRN_Y1, LRN_X_EDGES[0]:LRN_X_EDGES[-1]].copy()
    _add_time(timings, "crop", start)

    return aligned, crops


def split_lrn_cells(lrn_row):
    origin = LRN_X_EDGES[0]
    cells = []

    for index in range(12):
        left = LRN_X_EDGES[index] - origin + LRN_CELL_INSET_X
        right = LRN_X_EDGES[index + 1] - origin - LRN_CELL_INSET_X
        cells.append(lrn_row[LRN_CELL_INSET_Y:-LRN_CELL_INSET_Y, left:right])

    return cells


def draw_regions(aligned):
    visual = aligned.copy()

    for x, y, w, h in FIELD_BOXES.values():
        cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 0, 255), 4)

    for index in range(12):
        cv2.rectangle(
            visual,
            (LRN_X_EDGES[index], LRN_Y0),
            (LRN_X_EDGES[index + 1], LRN_Y1),
            (255, 0, 0),
            3,
        )

    return visual


def save_alignment_debug(image_path, aligned):
    ALIGNMENT_DEBUG_DIR.mkdir(exist_ok=True)
    visual = cv2.resize(draw_regions(aligned), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ALIGNMENT_DEBUG_DIR / f"{Path(image_path).stem}_aligned.png"), visual)


ENGINE_STEP_COUNT = 4

STATE_COLORS = {
    "pending": "#9e9e9e",
    "loading": "#d9a441",
    "ready": "#2e7d32",
    "error": "#c62828",
}


class EngineStatus:
    def __init__(self, name):
        self.name = name
        self.state = "pending"
        self.step = 0
        self.message = "Waiting..."

    def set(self, state, step, message):
        self.state = state
        self.step = step
        self.message = message


EASY_READER = None
TESS_READY = False
PADDLE_READER = None

EASY_NAME_READER = None
PADDLE_NAME_READER = None
TESS_NAME_LANG = "eng"
NAME_MODEL_INFO = {}
ENYE_SUPPORT = {}
ENYE_FAILURE_REASON = {}
TESS_NAME_EXTRA_CONFIG = ""
LOCAL_TESSDATA_DIR = Path(__file__).resolve().parent / "tessdata"


def _local_tessdata_args():
    folder = LOCAL_TESSDATA_DIR
    if (folder / "spa.traineddata").exists() and (folder / "eng.traineddata").exists():
        return f'--tessdata-dir "{folder.as_posix()}"'
    return ""


ENYE_VERIFIED = {}
ENYE_TEST_DETAILS = {}
ENYE_TEST_WORDS = ("Mu\u00f1oz", "Pe\u00f1a", "NU\u00d1EZ")


def _find_test_font(px):
    from PIL import ImageFont
    for name in ("arial.ttf", "Arial.ttf", "segoeui.ttf", "calibri.ttf", "LiberationSans-Regular.ttf",
                 "DejaVuSans.ttf", "Carlito-Regular.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return None


def _enye_test_images():
    from PIL import ImageDraw
    font = _find_test_font(56)
    if font is None:
        return []
    images = []
    for word in ENYE_TEST_WORDS:
        left, top, right, bottom = font.getbbox(word)
        canvas = Image.new("L", (right - left + 60, bottom - top + 40), 255)
        ImageDraw.Draw(canvas).text((30 - left, 20 - top), word, font=font, fill=0)
        images.append((word, np.array(canvas)))
    return images


def _test_text(details):
    return "; ".join(f"{word!r} was read as {got!r}" for word, got in details)


def enye_self_test(engine_name):
    images = _enye_test_images()
    if not images:
        return None, []

    details, hits = [], 0
    for word, image in images:
        try:
            read = run_engine_on_crop(engine_name, image, is_digit_cell=False)
        except Exception as error:
            read = f"<error: {type(error).__name__}>"
        details.append((word, read))
        if "\u00f1" in normalize(read):
            hits += 1
    return hits > 0, details


def _finish_enye_check(engine, report, loaded, not_loaded_message, suffix="", precomputed=None):
    if not loaded:
        ENYE_SUPPORT[engine] = False
        ENYE_VERIFIED[engine], ENYE_TEST_DETAILS[engine] = None, []
        report(4, not_loaded_message, "ready")
        return

    verified, details = precomputed if precomputed is not None else enye_self_test(engine)
    ENYE_VERIFIED[engine], ENYE_TEST_DETAILS[engine] = verified, details

    if verified is False:
        ENYE_SUPPORT[engine] = False
        _note_enye_failure(engine, "The Name model loaded, but the test words were read without an enye: "
                           + _test_text(details))
        report(4, "Ready, but a test image with an enye was read without it: the enye will NOT be read (see the log).",
               "ready")
    elif verified is True:
        report(4, "Ready.", "ready")
    else:
        report(4, "Ready (the Name reader could not be tested: no usable font).", "ready")


def _note_enye_failure(engine, reason):
    reason = " ".join(str(reason).split())[:400]
    ENYE_FAILURE_REASON[engine] = reason
    print(f"[{engine}] cannot read the enye: {reason}", file=sys.stderr)


def init_tesseract_engine(report):
    global TESS_READY, TESS_NAME_LANG, TESS_NAME_EXTRA_CONFIG

    report(1, "Looking for the tesseract program...", "loading")
    version = pytesseract.get_tesseract_version()

    try:
        installed_languages = set(pytesseract.get_languages(config=""))
    except Exception:
        installed_languages = set()

    has_spanish = "spa" in installed_languages
    used_local_folder = False
    TESS_NAME_EXTRA_CONFIG = ""

    if not has_spanish:
        local_args = _local_tessdata_args()
        if local_args:
            try:
                local_languages = set(pytesseract.get_languages(config=local_args))
            except Exception:
                local_languages = set()
            if "spa" in local_languages and "eng" in local_languages:
                has_spanish, used_local_folder, TESS_NAME_EXTRA_CONFIG = True, True, local_args

    TESS_NAME_LANG = "spa+eng" if has_spanish else "eng"
    NAME_MODEL_INFO["Tesseract"] = TESS_NAME_LANG + (" (local tessdata)" if used_local_folder else "")
    ENYE_SUPPORT["Tesseract"] = has_spanish
    if has_spanish:
        ENYE_FAILURE_REASON.pop("Tesseract", None)
    else:
        _note_enye_failure(
            "Tesseract",
            "Spanish data (spa.traineddata) was not found. Install it with Tesseract, or put "
            f"spa.traineddata and eng.traineddata in a folder named tessdata at {LOCAL_TESSDATA_DIR}",
        )

    report(2, f"Found tesseract version {version}.", "loading")

    report(3, "Running a warm-up OCR call...", "loading")
    blank = np.full((40, 120), 255, dtype=np.uint8)
    pytesseract.image_to_string(Image.fromarray(blank), config="--psm 7")
    name_config = "--psm 7" + (f" {TESS_NAME_EXTRA_CONFIG}" if TESS_NAME_EXTRA_CONFIG else "")
    pytesseract.image_to_string(Image.fromarray(blank), lang=TESS_NAME_LANG, config=name_config)

    TESS_READY = True
    _finish_enye_check(
        "Tesseract", report, has_spanish,
        "Ready, but Spanish data (spa) is missing: the enye will NOT be read (see the log).",
        suffix=", local tessdata folder" if used_local_folder else "",
    )


def init_easyocr_engine(report):
    global EASY_READER, EASY_NAME_READER

    report(1, "Importing EasyOCR...", "loading")

    report(2, "Building the EasyOCR reader (loading its models)...", "loading")
    EASY_READER = easyocr.Reader(["en"], gpu=False)

    report(3, "Building the Name reader, and warming up...", "loading")
    enye_ok = True
    try:
        EASY_NAME_READER = easyocr.Reader(["es", "en"], gpu=False)
        NAME_MODEL_INFO["EasyOCR"] = "es+en (latin model)"
    except Exception as error:
        EASY_NAME_READER = EASY_READER
        NAME_MODEL_INFO["EasyOCR"] = "en (no enye)"
        enye_ok = False
        _note_enye_failure("EasyOCR", f"{type(error).__name__}: {error}")

    ENYE_SUPPORT["EasyOCR"] = enye_ok

    blank = np.full((40, 120), 255, dtype=np.uint8)
    EASY_READER.recognize(blank, detail=0, paragraph=False)
    EASY_NAME_READER.recognize(blank, detail=0, paragraph=False)

    _finish_enye_check(
        "EasyOCR", report, enye_ok,
        "Ready, but the Spanish model could not be loaded: the enye will NOT be read (see the log).",
    )


def init_paddleocr_engine(report):
    global PADDLE_READER, PADDLE_NAME_READER

    report(1, "Importing PaddleOCR...", "loading")
    from paddleocr import PaddleOCR

    report(2, "Building the PaddleOCR pipelines (downloading models on first run)...", "loading")
    paddle_options = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    PADDLE_READER = PaddleOCR(lang="en", **paddle_options)

    attempts = [
        ("es (latin model)", {"lang": "es"}),
        ("es, PP-OCRv5", {"lang": "es", "ocr_version": "PP-OCRv5"}),
        ("es, PP-OCRv3", {"lang": "es", "ocr_version": "PP-OCRv3"}),
        ("latin", {"lang": "latin"}),
    ]
    enye_ok = False
    test_result = None
    failures = []
    report(3, "Building the Name pipeline, and warming up...", "loading")
    for label, options in attempts:
        try:
            candidate = PaddleOCR(**options, **paddle_options)
        except Exception as error:
            failures.append(f"{label}: {type(error).__name__}: {' '.join(str(error).split())[:200]}")
            continue

        PADDLE_NAME_READER = candidate
        NAME_MODEL_INFO["PaddleOCR"] = label
        verified, details = enye_self_test("PaddleOCR")
        if verified is False:
            failures.append(f"{label}: loaded, but the test words were read without an enye ({_test_text(details)})")
            continue

        enye_ok, test_result = True, (verified, details)
        ENYE_FAILURE_REASON.pop("PaddleOCR", None)
        break

    if not enye_ok:
        PADDLE_NAME_READER = PADDLE_READER
        NAME_MODEL_INFO["PaddleOCR"] = "en (no enye)"
        _note_enye_failure("PaddleOCR", " | ".join(failures))

    ENYE_SUPPORT["PaddleOCR"] = enye_ok

    blank = np.full((40, 120, 3), 255, dtype=np.uint8)
    PADDLE_READER.predict(blank)
    PADDLE_NAME_READER.predict(blank)

    _finish_enye_check(
        "PaddleOCR", report, enye_ok,
        "Ready, but no Latin/Spanish model could read the enye: the enye will NOT be read (see the log).",
        precomputed=test_result,
    )


DIGITS = "0123456789"


def _prepare_for_ocr(image, upscale=2, pad=20):
    gray = pipeline_raw(image)
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def run_tesseract(image, psm=7, whitelist=None, upscale=2, lang=None, extra_config=""):
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    if extra_config:
        config += f" {extra_config}"

    options = {"lang": lang} if lang else {}
    text = pytesseract.image_to_string(
        Image.fromarray(_prepare_for_ocr(image, upscale)), config=config, **options
    )
    return " ".join(text.split())


def run_easyocr(image, allowlist=None, upscale=2, reader=None):
    reader = reader or EASY_READER
    prepared = _prepare_for_ocr(image, upscale)

    try:
        result = reader.recognize(prepared, detail=0, paragraph=False, allowlist=allowlist)
    except Exception:
        result = reader.readtext(prepared, detail=0, paragraph=False, allowlist=allowlist)

    return " ".join(str(item).strip() for item in result).strip()


def _extract_paddle_texts(predict_result):
    texts = []

    for item in predict_result:
        rec_texts = None

        try:
            rec_texts = item["rec_texts"]
        except (TypeError, KeyError):
            rec_texts = getattr(item, "rec_texts", None)

        if rec_texts:
            texts.extend(str(text) for text in rec_texts)

    return texts


def run_paddleocr(image, upscale=2, reader=None):
    reader = reader or PADDLE_READER
    prepared = _prepare_for_ocr(image, upscale)
    color = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

    result = reader.predict(color)
    texts = _extract_paddle_texts(result)

    return " ".join(text.strip() for text in texts).strip()


def run_engine_on_crop(engine_name, crop, is_digit_cell):
    if engine_name == "Tesseract":
        if is_digit_cell:
            return run_tesseract(crop, psm=10, whitelist=DIGITS, upscale=3)
        return run_tesseract(crop, psm=7, lang=TESS_NAME_LANG, extra_config=TESS_NAME_EXTRA_CONFIG)

    if engine_name == "EasyOCR":
        if is_digit_cell:
            return run_easyocr(crop, allowlist=DIGITS, upscale=3)
        return run_easyocr(crop, reader=EASY_NAME_READER)

    if engine_name == "PaddleOCR":
        if is_digit_cell:
            return run_paddleocr(crop, upscale=3)
        return run_paddleocr(crop, reader=PADDLE_NAME_READER)

    raise ValueError(f"unknown engine: {engine_name}")


def read_fields(processed, engine_name, timings=None):
    start = time.perf_counter()
    name = restrict_to_enye(run_engine_on_crop(engine_name, processed["name"], is_digit_cell=False))
    _add_time(timings, "ocr_name", start)

    start = time.perf_counter()
    digits = [
        run_engine_on_crop(engine_name, cell, is_digit_cell=True)
        for cell in split_lrn_cells(processed["lrn_row"])
    ]
    lrn = "".join((digit.replace(" ", "")[:1] or "?") for digit in digits)
    _add_time(timings, "ocr_lrn", start)

    return name, lrn


def evaluate_prediction(gt_fields, prediction):
    name_gt = normalize(gt_fields[0])
    lrn_gt = normalize_lrn(gt_fields[1])
    name_pred = normalize(prediction[0])
    lrn_pred = normalize_lrn(prediction[1])

    name_cer = field_cer(name_gt, name_pred)
    lrn_cer = field_cer(lrn_gt, lrn_pred)
    pooled = pooled_cer((name_gt, lrn_gt), (name_pred, lrn_pred))

    prediction_is_empty = not name_pred and set(lrn_pred) <= {"?"}

    if not (name_gt or lrn_gt):
        status = "invalid_ground_truth"
    elif prediction_is_empty:
        status = "empty_prediction"
    else:
        status = "success"

    return {
        "name_cer": name_cer,
        "lrn_cer": lrn_cer,
        "pooled_cer": pooled,
        "status": status,
        "pred_name": prediction[0],
        "pred_lrn": prediction[1],
        "gt_name_norm": name_gt,
        "gt_lrn_norm": lrn_gt,
        "pred_name_norm": name_pred,
        "pred_lrn_norm": lrn_pred,
        "raw_prediction": f"NAME: {prediction[0]}\nLRN: {prediction[1]}",
    }


class QueueItem:
    def __init__(self, path, image, load_index):
        self.path = path
        self.image_id = Path(path).stem
        self.align_times = {}
        self.image = image
        self.load_index = load_index

        self.aligned = None
        self.crops = None
        self.align_error = None

        self.ground_truth = ("", "")
        self.status = "pending"
        self.status_detail = ""

        self.preview_thumbnail = None

        self.checked = tk.BooleanVar(value=False)

    def display_text(self):
        label = f"{self.image_id}"

        if self.status == "pending":
            return f"{label} - Pending"
        if self.status == "processing":
            return f"{label} - Processing..."
        if self.status == "done":
            return f"{label} - Done"
        if self.status == "ocr_failed":
            return f"{label} - Did not read by OCR"
        if self.status == "error":
            return f"{label} - Error: {self.status_detail}"
        if self.status == "skipped":
            return f"{label} - Skipped (no corner marks found)"
        return label


def _to_float(value):
    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        if value in (None, "", "NA"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def combo_key(row):
    preprocessing = row.get("preprocessing", "")
    strength = _to_float(row.get("strength"))
    block = _to_int(row.get("block"))
    c_value = _to_int(row.get("C"))

    if preprocessing == "raw" or strength is None or block is None or c_value is None:
        return (preprocessing, "NA", "NA", "NA")
    return (preprocessing, strength, block, c_value)


def combo_label(key):
    preprocessing, strength, block, c_value = key
    if preprocessing == "raw":
        return "raw"
    return f"strength={strength}, block={block}, C={c_value}"


def aggregate_results(rows):
    groups = {}

    for row in rows:
        pooled = _to_float(row.get("pooled_cer"))
        if pooled is None:
            continue

        engine = row.get("engine", "")
        key = combo_key(row)
        entry = groups.setdefault(key, {}).setdefault(engine, {"pooled": [], "ocr": []})
        entry["pooled"].append(pooled)

        ocr_seconds = _to_float(row.get("ocr_seconds"))
        if ocr_seconds is not None:
            entry["ocr"].append(ocr_seconds)

    aggregated = {}
    for key, per_engine in groups.items():
        aggregated[key] = {}
        for engine, values in per_engine.items():
            aggregated[key][engine] = {
                "pooled_mean": statistics.mean(values["pooled"]),
                "ocr_mean": statistics.mean(values["ocr"]) if values["ocr"] else None,
                "count": len(values["pooled"]),
            }

    return aggregated


def combo_time_totals(rows):
    totals = {}

    for row in rows:
        ocr_seconds = _to_float(row.get("ocr_seconds"))
        preprocess_seconds = _to_float(row.get("preprocess_seconds"))
        if ocr_seconds is None and preprocess_seconds is None:
            continue

        key = combo_key(row)
        entry = totals.setdefault(key, {"seconds": 0.0, "images": set()})
        entry["seconds"] += (ocr_seconds or 0.0) + (preprocess_seconds or 0.0)
        entry["images"].add(row.get("image_id", ""))

    return {
        key: {"seconds": entry["seconds"], "images": len(entry["images"])}
        for key, entry in totals.items()
    }


STAGE_COLUMNS = [
    "stage_align_seconds", "stage_crop_seconds",
    "stage_grayscale_seconds", "stage_denoise_seconds",
    "stage_sharpen_seconds", "stage_threshold_seconds",
    "stage_ocr_name_seconds", "stage_ocr_lrn_seconds", "stage_score_seconds",
]

SHARED_STAGE_ROWS = [
    ("stage_align_seconds", "Alignment (corner marks + page warp)", "once per image"),
    ("stage_crop_seconds", "Field cropping", "once per image"),
    ("stage_grayscale_seconds", "Preprocess: grayscale", "per image & setting"),
    ("stage_denoise_seconds", "Preprocess: denoise (Non-Local Means)", "per image & setting"),
    ("stage_sharpen_seconds", "Preprocess: sharpen", "per image & setting"),
    ("stage_threshold_seconds", "Preprocess: adaptive threshold", "per image & setting"),
]


def format_stage_time(seconds):
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def _mean_column(rows, column):
    values = [_to_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def stage_timing_stats(rows, engine_names=ENGINE_NAMES):
    shared = {column: _mean_column(rows, column) for column, _label, _scope in SHARED_STAGE_ROWS}

    processed_rows = [row for row in rows if _to_float(row.get("stage_denoise_seconds")) is not None]
    shared["preprocess_total"] = _mean_column(processed_rows, "preprocess_seconds")

    engines = {}
    for engine in engine_names:
        engine_rows = [
            row for row in rows
            if row.get("engine") == engine and _to_float(row.get("stage_ocr_name_seconds")) is not None
        ]
        entry = {
            "ocr_name": _mean_column(engine_rows, "stage_ocr_name_seconds"),
            "ocr_lrn": _mean_column(engine_rows, "stage_ocr_lrn_seconds"),
            "ocr_total": _mean_column(engine_rows, "ocr_seconds"),
            "score": _mean_column(engine_rows, "stage_score_seconds"),
            "tasks": len(engine_rows),
        }
        if entry["ocr_total"] is None:
            entry["total"] = None
        else:
            parts = [shared["preprocess_total"], entry["ocr_total"], entry["score"]]
            entry["total"] = sum(part for part in parts if part is not None)
        engines[engine] = entry

    return {"shared": shared, "engines": engines}


def timing_table_rows(stats, engine_names=ENGINE_NAMES):
    table = []

    for column, label, scope in SHARED_STAGE_ROWS:
        table.append((label, scope, stats["shared"].get(column), {}))
    table.append(("Preprocessing total (denoise + sharpen + threshold settings)",
                  "per image & setting", stats["shared"].get("preprocess_total"), {}))

    engine_rows = [
        ("ocr_name", "OCR: Name field"),
        ("ocr_lrn", "OCR: LRN (12 digit cells)"),
        ("ocr_total", "OCR total"),
        ("score", "Scoring (normalize + CER)"),
        ("total", "Total per image & setting (preprocessing + OCR + scoring)"),
    ]
    for key, label in engine_rows:
        table.append((label, "per image, setting & engine", None,
                      {engine: stats["engines"][engine][key] for engine in engine_names}))

    return table


def best_combo_per_engine(aggregated, engine_names=ENGINE_NAMES):
    best = {}

    for engine in engine_names:
        candidates = [
            (key, per_engine[engine]["pooled_mean"])
            for key, per_engine in aggregated.items()
            if engine in per_engine and per_engine[engine]["pooled_mean"] is not None
        ]
        if candidates:
            best[engine] = min(candidates, key=lambda pair: pair[1])

    return best


def best_combo_overall(aggregated, engine_names=ENGINE_NAMES):
    candidates = []

    for key, per_engine in aggregated.items():
        values = [
            per_engine[engine]["pooled_mean"]
            for engine in engine_names
            if engine in per_engine and per_engine[engine]["pooled_mean"] is not None
        ]
        if values:
            candidates.append((key, statistics.mean(values)))

    if not candidates:
        return None

    return min(candidates, key=lambda pair: pair[1])


def overall_engine_stats(rows, engine_names=ENGINE_NAMES):
    stats = {}

    for engine in engine_names:
        engine_rows = [row for row in rows if row.get("engine") == engine]
        pooled_values = [
            _to_float(row.get("pooled_cer"))
            for row in engine_rows
            if _to_float(row.get("pooled_cer")) is not None
        ]
        ocr_values = [
            _to_float(row.get("ocr_seconds"))
            for row in engine_rows
            if _to_float(row.get("ocr_seconds")) is not None
        ]
        statuses = [row.get("status", "") for row in engine_rows]

        stats[engine] = {
            "mean": statistics.mean(pooled_values) if pooled_values else None,
            "median": statistics.median(pooled_values) if pooled_values else None,
            "stdev": statistics.pstdev(pooled_values) if len(pooled_values) > 1 else None,
            "mean_ocr_seconds": statistics.mean(ocr_values) if ocr_values else None,
            "n": len(pooled_values),
            "success": statuses.count("success"),
            "empty_prediction": statuses.count("empty_prediction"),
            "invalid_ground_truth": statuses.count("invalid_ground_truth"),
        }

    return stats


class ImageChecklist(ttk.Frame):
    def __init__(self, parent, on_row_clicked, canvas_bg="#1b2735", selected_bg="#24405c"):
        super().__init__(parent, style="Card.TFrame")
        self.on_row_clicked = on_row_clicked
        self.row_widgets = {}
        self.selected_item = None
        self.selected_bg = selected_bg

        self.canvas = tk.Canvas(self, highlightthickness=0, background=canvas_bg)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame")

        self.inner.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width),
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def rebuild(self, items):
        for child in self.inner.winfo_children():
            child.destroy()
        self.row_widgets = {}

        for item in items:
            row = ttk.Frame(self.inner, style="Card.TFrame")
            row.pack(fill="x", pady=1)

            check = ttk.Checkbutton(row, variable=item.checked, style="Card.TCheckbutton")
            check.pack(side="left", padx=(2, 4))

            label = ttk.Label(row, text=item.display_text(), anchor="w", style="Card.TLabel")
            label.pack(side="left", fill="x", expand=True)
            label.bind("<Button-1>", lambda event, image_id=item.image_id: self.on_row_clicked(image_id))

            if item.image_id == self.selected_item:
                label.configure(background=self.selected_bg)

            self.row_widgets[item.image_id] = (row, label)

    def set_selected(self, image_id):
        self.selected_item = image_id
        for current_id, (_row, label) in self.row_widgets.items():
            label.configure(background=self.selected_bg if current_id == image_id else "")

    def update_row_text(self, item):
        pair = self.row_widgets.get(item.image_id)
        if pair:
            pair[1].configure(text=item.display_text())


class OCRExperimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resumable OCR CER Experiment")

        self.root.minsize(1000, 650)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                screen_w = self.root.winfo_screenwidth()
                screen_h = self.root.winfo_screenheight()
                self.root.geometry(f"{min(1650, screen_w - 50)}x{min(950, screen_h - 80)}")

        self.queue = []
        self.next_load_index = 0
        self.results = []
        self.raw_baselines = {}
        self.completed_keys = set()
        self.completed_image_ids = set()
        self.last_results_path = None
        self.stop_requested = False
        self.abort_requested = False
        self.running = False
        self.current_preview_image = None
        self._preview_source_image = None
        self._preview_redraw_job = None
        self._last_auto_previewed_id = None

        self.engine_status = {name: EngineStatus(name) for name in ENGINE_NAMES}
        self.engine_badge_widgets = {}

        self.cumulative_rows = []
        self.overwritten_image_ids = set()
        self.load_cumulative_csv()

        self.apply_theme()
        self.build_gui()
        self.start_engine_loading()


    def apply_theme(self):
        self.colors = {
            "bg": "#121a24",
            "card": "#1b2735",
            "card_border": "#2b3b4e",
            "text": "#e6edf3",
            "text_dim": "#9fb0c0",
            "accent": "#3b82f6",
            "accent_dark": "#2a5db0",
            "entry_bg": "#0f1824",
            "row_selected": "#24405c",
        }
        colors = self.colors

        self.root.configure(background=colors["bg"])

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=colors["bg"], foreground=colors["text"],
                         fieldbackground=colors["entry_bg"])
        style.configure("TFrame", background=colors["bg"])
        style.configure("TLabel", background=colors["bg"], foreground=colors["text"])
        style.configure("TCheckbutton", background=colors["bg"], foreground=colors["text"])
        style.map("TCheckbutton", background=[("active", colors["bg"])])

        style.configure("TLabelframe", background=colors["card"], bordercolor=colors["card_border"],
                         relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=colors["card"], foreground=colors["text"],
                         font=("TkDefaultFont", 10, "bold"))

        style.configure("TButton", background=colors["card"], foreground=colors["text"],
                         bordercolor=colors["card_border"], focusthickness=1, padding=5)
        style.map("TButton",
                  background=[("active", colors["accent_dark"]), ("pressed", colors["accent_dark"])],
                  foreground=[("disabled", colors["text_dim"])])

        style.configure("TEntry", fieldbackground=colors["entry_bg"], foreground=colors["text"],
                         bordercolor=colors["card_border"], insertcolor=colors["text"])

        style.configure("TNotebook", background=colors["bg"], bordercolor=colors["card_border"])
        style.configure("TNotebook.Tab", background=colors["card"], foreground=colors["text_dim"],
                         padding=(14, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", colors["accent"])],
                  foreground=[("selected", "#ffffff")])

        style.configure("TPanedwindow", background=colors["bg"])

        style.configure("Vertical.TScrollbar", background=colors["card"], troughcolor=colors["bg"],
                         bordercolor=colors["bg"], arrowcolor=colors["text_dim"])
        style.configure("Horizontal.TScrollbar", background=colors["card"], troughcolor=colors["bg"],
                         bordercolor=colors["bg"], arrowcolor=colors["text_dim"])

        style.configure("TScale", background=colors["card"], troughcolor=colors["entry_bg"])

        style.configure("Treeview", background=colors["entry_bg"], fieldbackground=colors["entry_bg"],
                         foreground=colors["text"], bordercolor=colors["card_border"], rowheight=22)
        style.configure("Treeview.Heading", background=colors["card"], foreground=colors["text"],
                         bordercolor=colors["card_border"])
        style.map("Treeview", background=[("selected", colors["accent_dark"])])

        style.configure("TProgressbar", background=colors["accent"], troughcolor=colors["entry_bg"],
                         bordercolor=colors["card_border"])
        style.configure("Accent.Horizontal.TProgressbar", background=colors["accent"],
                         troughcolor=colors["entry_bg"], bordercolor=colors["card_border"],
                         thickness=18)

        style.configure("Card.TFrame", background=colors["card"])
        style.configure("Card.TLabel", background=colors["card"], foreground=colors["text"])
        style.configure("Card.TCheckbutton", background=colors["card"], foreground=colors["text"])
        style.map("Card.TCheckbutton", background=[("active", colors["card"])])

    def text_widget_colors(self):
        colors = self.colors
        return {
            "background": colors["entry_bg"],
            "foreground": colors["text"],
            "insertbackground": colors["text"],
            "highlightthickness": 0,
        }


    def build_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.process_tab = ttk.Frame(self.notebook)
        self.single_run_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.process_tab, text="Process")
        self.notebook.add(self.single_run_tab, text="Single Run")
        self.notebook.add(self.results_tab, text="Results")

        self.build_process_tab()
        self.build_single_run_tab()
        self.build_results_tab()

    def build_process_tab(self):
        root = self.process_tab

        status_row = ttk.Frame(root, padding=(8, 8, 8, 4))
        status_row.pack(fill="x")

        ttk.Label(status_row, text="OCR Status:").pack(side="left", padx=(0, 10))

        for name in ENGINE_NAMES:
            self.engine_badge_widgets[name] = self.build_engine_badge(status_row, name)

        self.progress_percent_var = tk.StringVar(value="0%")
        ttk.Label(
            status_row, textvariable=self.progress_percent_var, width=5,
            font=("TkDefaultFont", 10, "bold"), foreground=self.colors["accent"],
        ).pack(side="right", padx=(0, 10))

        self.progress = ttk.Progressbar(
            status_row, length=320, mode="determinate", style="Accent.Horizontal.TProgressbar",
        )
        self.progress.pack(side="right", padx=10)
        ttk.Label(status_row, text="Overall progress:").pack(side="right")

        remarks_frame = ttk.Frame(root, padding=(8, 4))
        ttk.Label(
            remarks_frame, text="Remarks", background="#2e7d32", foreground="white",
            anchor="center", padding=4,
        ).pack(fill="x")
        ttk.Label(
            remarks_frame,
            text="Cannot start if no ground truth is entered. One line per image: NAME | LRN "
                 "(Surname First Name, then the 12-digit LRN), in the order the images were loaded.",
        ).pack(anchor="w", pady=(2, 0))

        bottom = ttk.Frame(root, padding=8)

        left_controls = ttk.Frame(bottom)
        left_controls.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(left_controls, text="Raw image name:").pack(anchor="w")
        self.raw_name_var = tk.StringVar(value="")
        ttk.Entry(left_controls, textvariable=self.raw_name_var, state="readonly", width=28).pack(anchor="w")

        ttk.Label(left_controls, text="Status:").pack(anchor="w", pady=(6, 0))
        self.status_var = tk.StringVar(value="Select images first.")
        ttk.Entry(left_controls, textvariable=self.status_var, state="readonly", width=28).pack(anchor="w")

        button_row = ttk.Frame(left_controls)
        button_row.pack(anchor="w", pady=(8, 0))
        ttk.Button(button_row, text="Select Images", command=self.select_images).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(button_row, text="Load Checkpoint", command=self.load_checkpoint_dialog).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(button_row, text="Start", command=self.start_experiment).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(button_row, text="Remove current", command=self.remove_current_image).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(button_row, text="Pause", command=self.request_pause).grid(row=2, column=0, padx=2, pady=2)

        abort_button = tk.Button(
            button_row, text="Abort", command=self.request_abort,
            background="#c62828", foreground="white", activebackground="#b71c1c",
        )
        abort_button.grid(row=2, column=1, padx=2, pady=2, sticky="ew")

        gt_frame = ttk.LabelFrame(bottom, text="Ground Truth")
        gt_frame.pack(side="left", fill="both", expand=True)
        self.gt_box = scrolledtext.ScrolledText(gt_frame, height=8, wrap=tk.NONE, **self.text_widget_colors())
        self.gt_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.build_best_observed_panel(bottom)

        bottom.pack(side="bottom", fill="x")
        remarks_frame.pack(side="bottom", fill="x")

        panes = ttk.PanedWindow(root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=4)

        preview_frame = ttk.LabelFrame(panes, text="Image Preview")
        middle_frame = ttk.LabelFrame(panes, text="Processed Images")
        log_frame = ttk.LabelFrame(panes, text="Results / Checkpoint Log")

        panes.add(preview_frame, weight=2)
        panes.add(middle_frame, weight=3)
        panes.add(log_frame, weight=2)

        self.preview_label = ttk.Label(
            preview_frame, text="Select an image to preview it here.", anchor="center", style="Card.TLabel",
        )
        self.preview_label.pack(fill="both", expand=True, padx=4, pady=4)
        preview_frame.bind("<Configure>", lambda event: self.redraw_preview())

        self.checklist = ImageChecklist(
            middle_frame, on_row_clicked=self.select_preview_image,
            canvas_bg=self.colors["card"], selected_bg=self.colors["row_selected"],
        )
        self.checklist.pack(fill="both", expand=True, padx=4, pady=4)

        middle_buttons = ttk.Frame(middle_frame, style="Card.TFrame")
        middle_buttons.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(middle_buttons, text="Delete checked", command=self.delete_checked).pack(side="left", padx=2)
        ttk.Button(middle_buttons, text="Move up", command=lambda: self.move_checked(-1)).pack(side="left", padx=2)
        ttk.Button(middle_buttons, text="Move down", command=lambda: self.move_checked(1)).pack(side="left", padx=2)
        ttk.Button(middle_buttons, text="Save results (CSV/JSON)", command=self.save_results_file).pack(side="left", padx=2)

        self.log_box = scrolledtext.ScrolledText(log_frame, wrap="word", state="disabled", **self.text_widget_colors())
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.refresh_best_observed()

    def build_engine_badge(self, parent, name):
        frame = ttk.Frame(parent, padding=4)
        frame.pack(side="left", padx=8)

        state_label = tk.Label(
            frame, text=name, background=STATE_COLORS["pending"], foreground="white",
            padx=10, pady=4, font=("TkDefaultFont", 10, "bold"),
        )
        state_label.pack()

        squares_frame = ttk.Frame(frame)
        squares_frame.pack(pady=(4, 0))

        squares = []
        for _ in range(ENGINE_STEP_COUNT):
            square = tk.Frame(squares_frame, width=18, height=14, background=STATE_COLORS["pending"])
            square.pack(side="left", padx=1)
            squares.append(square)

        message_label = ttk.Label(
            frame, text="Waiting...", font=("TkDefaultFont", 8), wraplength=190, justify="center",
        )
        message_label.pack()

        enye_label = tk.Label(
            frame, text="\u00f1: waiting...", background=STATE_COLORS["pending"], foreground="white",
            padx=6, pady=1, font=("TkDefaultFont", 8, "bold"),
        )

        return {"state_label": state_label, "squares": squares,
                "message_label": message_label, "enye_label": enye_label}

    def build_best_observed_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Best Observed Preprocessing")
        frame.pack(side="left", fill="y", padx=(8, 0))

        columns = ["engine", "strength", "block", "c", "pooled_cer", "scan_time"]
        headings = {
            "engine": "OCR Engine", "strength": "Strength", "block": "Block",
            "c": "C", "pooled_cer": "Best Pooled CER", "scan_time": "Scan Time / Img (s)",
        }

        self.best_observed_tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=len(ENGINE_NAMES),
        )
        for column in columns:
            self.best_observed_tree.heading(column, text=headings[column])
            self.best_observed_tree.column(column, width=120 if column == "scan_time" else 90, anchor="center")
        self.best_observed_tree.pack(padx=4, pady=(4, 0))

        for engine in ENGINE_NAMES:
            self.best_observed_tree.insert("", "end", iid=engine, values=[engine, "-", "-", "-", "-", "-"])

        ttk.Button(frame, text="Refresh", command=self.refresh_best_observed).pack(pady=4)

    def refresh_best_observed(self):
        aggregated = aggregate_results(self.cumulative_rows)
        best = best_combo_per_engine(aggregated)

        for engine in ENGINE_NAMES:
            if engine in best:
                key, pooled_mean = best[engine]
                preprocessing, strength, block, c_value = key
                ocr_mean = aggregated[key][engine]["ocr_mean"]
                time_text = "-" if ocr_mean is None else f"{ocr_mean:.2f}"

                if preprocessing == "raw":
                    values = [engine, "raw", "raw", "raw", f"{pooled_mean:.4f}", time_text]
                else:
                    values = [engine, strength, block, c_value, f"{pooled_mean:.4f}", time_text]
            else:
                values = [engine, "-", "-", "-", "-", "-"]

            self.best_observed_tree.item(engine, values=values)

    def build_single_run_tab(self):
        root = self.single_run_tab

        ttk.Label(
            root,
            text="Run one exact combination of settings on the currently loaded images, "
                 "instead of the full 100-setting grid search. Only the same values the "
                 "grid search itself uses are offered here.",
            padding=8,
        ).pack(fill="x")

        card = ttk.LabelFrame(root, text="Custom Parameters", padding=12)
        card.pack(fill="x", padx=8, pady=4)

        self.custom_strength_var = tk.StringVar(value=str(STRENGTHS[len(STRENGTHS) // 2]))
        self.custom_block_var = tk.StringVar(value=str(BLOCKS[len(BLOCKS) // 2]))
        self.custom_c_var = tk.StringVar(value=str(C_VALUES[len(C_VALUES) // 2]))

        self.build_value_picker(card, "Strength", self.custom_strength_var, STRENGTHS)
        self.build_value_picker(card, "Block size", self.custom_block_var, BLOCKS)
        self.build_value_picker(card, "C", self.custom_c_var, C_VALUES)

        ttk.Button(card, text="Run this combination", command=self.start_single_run).pack(
            anchor="w", pady=(10, 0)
        )

        ttk.Label(
            root,
            text="Progress and results appear in the Process tab's log, and this setting "
                 "will show up as its own row in the Results tab once it has run.",
            padding=8,
        ).pack(fill="x")

    def build_value_picker(self, parent, label_text, variable, allowed_values):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)

        ttk.Label(row, text=label_text, width=12).pack(side="left")

        combo = ttk.Combobox(
            row, textvariable=variable, values=[str(value) for value in allowed_values],
            state="readonly", width=10,
        )
        combo.pack(side="left", padx=10)

        ttk.Label(row, text=f"(one of the {len(allowed_values)} values the grid search uses)").pack(side="left")

    def build_results_tab(self):
        root = self.results_tab
        self.combo_report_rows = []

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Refresh statistics", command=self.refresh_results_tab).pack(side="left")
        ttk.Button(top, text="Export combo report", command=self.export_combo_report).pack(side="left", padx=(8, 0))
        ttk.Label(
            top,
            text="Statistics are computed from every result on file, across every run, not just this session.",
        ).pack(side="left", padx=10)

        summary_frame = ttk.LabelFrame(root, text="Summary", padding=8)
        summary_frame.pack(fill="x", padx=8, pady=4)
        self.summary_text = tk.Text(summary_frame, height=14, wrap="word", state="disabled", **self.text_widget_colors())
        self.summary_text.pack(fill="x")

        table_notebook = ttk.Notebook(root)
        table_notebook.pack(fill="both", expand=True, padx=8, pady=4)

        matrix_frame = ttk.Frame(table_notebook, padding=4)
        table_notebook.add(matrix_frame, text="Preprocessing Setting Comparison")

        timing_frame = ttk.Frame(table_notebook, padding=4)
        table_notebook.add(timing_frame, text="Timing by Engine and Stage")
        self.build_timing_table(timing_frame)

        columns = ["setting", "n_images"]
        for engine in ENGINE_NAMES:
            columns += [f"{engine}_cer", f"{engine}_time"]
        columns += ["total_time", "time_per_image"]

        self.matrix_tree = ttk.Treeview(matrix_frame, columns=columns, show="headings")
        headings = {
            "setting": "Preprocessing setting", "n_images": "Images",
            "total_time": "Combo total time", "time_per_image": "Time / image (all engines)",
        }
        for engine in ENGINE_NAMES:
            headings[f"{engine}_cer"] = f"{engine} pooled CER"
            headings[f"{engine}_time"] = f"{engine} OCR s/img"

        for column in columns:
            self.matrix_tree.heading(column, text=headings.get(column, column))
            if column == "setting":
                width = 220
            elif column in ("total_time", "time_per_image"):
                width = 140
            else:
                width = 110
            self.matrix_tree.column(column, width=width, anchor="center")

        y_scroll = ttk.Scrollbar(matrix_frame, orient="vertical", command=self.matrix_tree.yview)
        x_scroll = ttk.Scrollbar(matrix_frame, orient="horizontal", command=self.matrix_tree.xview)
        self.matrix_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        self.matrix_tree.pack(side="left", fill="both", expand=True)


    def start_engine_loading(self):
        loaders = {
            "Tesseract": init_tesseract_engine,
            "EasyOCR": init_easyocr_engine,
            "PaddleOCR": init_paddleocr_engine,
        }

        for name, loader in loaders.items():
            thread = threading.Thread(target=self.run_engine_loader, args=(name, loader), daemon=True)
            thread.start()

    def run_engine_loader(self, name, loader):
        def report(step, message, state):
            self.root.after(0, lambda: self.update_engine_badge(name, state, step, message))

        try:
            loader(report)
        except Exception as error:
            message = str(error)
            self.root.after(0, lambda: self.update_engine_badge(name, "error", 0, message))
            self.root.after(0, lambda: self.log(f"{name} failed to load: {message}"))

    def update_engine_badge(self, name, state, step, message):
        status = self.engine_status[name]
        status.set(state, step, message)

        widgets = self.engine_badge_widgets[name]
        widgets["state_label"].configure(background=STATE_COLORS[state])
        widgets["message_label"].configure(text=message)
        self.update_enye_indicator(name, state)

        for index, square in enumerate(widgets["squares"]):
            if state == "error":
                color = STATE_COLORS["error"] if index < ENGINE_STEP_COUNT else STATE_COLORS["pending"]
            elif index < step:
                color = STATE_COLORS["ready"] if state == "ready" else STATE_COLORS["loading"]
            else:
                color = STATE_COLORS["pending"]
            square.configure(background=color)

        if state == "ready":
            supported = ENYE_SUPPORT.get(name)
            model = NAME_MODEL_INFO.get(name, "")
            if supported is True:
                if ENYE_VERIFIED.get(name) is True:
                    self.log(f"{name} is ready.")
                else:
                    self.log(f"{name} is ready. Its Name model loaded but could not be tested (Name model: {model}).")
            elif supported is False:
                reason = ENYE_FAILURE_REASON.get(name, "reason not recorded")
                self.log(
                    f"WARNING: {name} is ready but will NOT read the enye (Name model: {model}). "
                    f"Reason: {reason}"
                )
            else:
                self.log(f"{name} is ready.")

    def update_enye_indicator(self, name, state):
        label = self.engine_badge_widgets[name]["enye_label"]
        model = NAME_MODEL_INFO.get(name, "")
        text = color = None

        if state == "ready":
            supported = ENYE_SUPPORT.get(name)
            verified = ENYE_VERIFIED.get(name)
            if supported is True and verified is True:
                pass
            elif supported is True:
                text, color = f"\u00f1 loaded, not verified ({model})", STATE_COLORS["loading"]
            elif supported is False:
                text, color = f"\u00f1 NOT read ({model})", STATE_COLORS["error"]

        if text is None:
            label.pack_forget()
        else:
            label.configure(text=text, background=color)
            if not label.winfo_ismapped():
                label.pack(pady=(3, 0))

    def engine_ready(self, name):
        return self.engine_status[name].state == "ready"


    _LOG_MAX_LINES = 2000
    _LOG_TRIM_TO = 1500

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {message}\n")

        line_count = int(self.log_box.index("end-1c").split(".")[0])
        if line_count > self._LOG_MAX_LINES:
            cut_at = line_count - self._LOG_TRIM_TO
            self.log_box.delete("1.0", f"{cut_at}.0")

        self.log_box.see("end")
        self.log_box.configure(state="disabled")


    def select_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        alignment_failures = []

        for path in paths:
            image = cv2.imread(path)
            if image is None:
                messagebox.showwarning("Image error", f"Could not read image:\n{path}")
                continue

            item = QueueItem(path, image, self.next_load_index)
            self.next_load_index += 1

            try:
                load_timings = {}
                aligned, crops = extract_field_crops(image, timings=load_timings)
                item.align_times = {
                    "stage_align_seconds": load_timings.get("align"),
                    "stage_crop_seconds": load_timings.get("crop"),
                }
                item.aligned = aligned
                item.crops = crops
                if SAVE_ALIGNMENT_DEBUG:
                    save_alignment_debug(path, aligned)
            except Exception as error:
                item.status = "skipped"
                item.align_error = str(error)
                alignment_failures.append(f"{item.image_id}: {error}")

            item.preview_thumbnail = self.make_preview_thumbnail(item)

            self.queue.append(item)

        if alignment_failures:
            messagebox.showwarning(
                "Corner marks not found",
                "These images could not be aligned and will be skipped:\n\n"
                + "\n".join(alignment_failures),
            )

        self.checklist.rebuild(self.queue)
        aligned_count = sum(item.crops is not None for item in self.queue)
        self.status_var.set(f"Loaded {len(self.queue)} image(s); {aligned_count} aligned.")
        self.log(f"Loaded {len(paths)} image(s) from disk.")

        if self.checklist.selected_item is None:
            first_aligned = next((item for item in self.queue if item.crops is not None), None)
            first_item = first_aligned or (self.queue[0] if self.queue else None)
            if first_item is not None:
                self.select_preview_image(first_item.image_id)

    def select_preview_image(self, image_id):
        item = self.find_item(image_id)
        if item is None:
            return

        self.checklist.set_selected(image_id)
        self.raw_name_var.set(item.image_id)
        self.status_var.set(item.display_text())
        self.show_preview(item)

    def find_item(self, image_id):
        for item in self.queue:
            if item.image_id == image_id:
                return item
        return None

    _PREVIEW_SOURCE_MAX_DIM = 1200

    def make_preview_thumbnail(self, item):
        source = draw_regions(item.aligned) if item.aligned is not None else item.image
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((self._PREVIEW_SOURCE_MAX_DIM, self._PREVIEW_SOURCE_MAX_DIM))
        return pil_image

    def show_preview(self, item):
        if item.preview_thumbnail is None:
            item.preview_thumbnail = self.make_preview_thumbnail(item)

        self._preview_source_image = item.preview_thumbnail
        self.redraw_preview()

    def redraw_preview(self):
        if getattr(self, "_preview_redraw_job", None) is not None:
            self.root.after_cancel(self._preview_redraw_job)
        self._preview_redraw_job = self.root.after(120, self._redraw_preview_now)

    def _redraw_preview_now(self):
        self._preview_redraw_job = None

        pil_source = getattr(self, "_preview_source_image", None)
        if pil_source is None:
            return

        panel_w = max(self.preview_label.winfo_width(), 100)
        panel_h = max(self.preview_label.winfo_height(), 100)

        pil_image = pil_source.copy()
        pil_image.thumbnail((panel_w - 10, panel_h - 10))

        self.current_preview_image = ImageTk.PhotoImage(pil_image)
        self.preview_label.configure(image=self.current_preview_image, text="")

    def clear_preview(self):
        if getattr(self, "_preview_redraw_job", None) is not None:
            self.root.after_cancel(self._preview_redraw_job)
            self._preview_redraw_job = None

        self._preview_source_image = None
        self.current_preview_image = None
        self.preview_label.configure(image="", text="Select an image to preview it here.")
        self.checklist.selected_item = None

    def delete_checked(self):
        if self.running:
            messagebox.showinfo("Experiment running", "Pause or abort the experiment before editing the queue.")
            return

        checked = [item for item in self.queue if item.checked.get()]
        if not checked:
            messagebox.showinfo("Nothing checked", "Check one or more images first.")
            return

        checked_ids = {item.image_id for item in checked}

        if self.checklist.selected_item in checked_ids:
            self.clear_preview()
            self.raw_name_var.set("")
            self.status_var.set("Removed.")

        self.queue = [item for item in self.queue if item.image_id not in checked_ids]
        self.checklist.rebuild(self.queue)
        self.log(f"Removed {len(checked)} image(s) from the queue.")

    def move_checked(self, direction):
        if self.running:
            messagebox.showinfo("Experiment running", "Pause or abort the experiment before editing the queue.")
            return

        indices = [index for index, item in enumerate(self.queue) if item.checked.get()]
        if not indices:
            return

        if direction < 0:
            indices = sorted(indices)
        else:
            indices = sorted(indices, reverse=True)

        for index in indices:
            new_index = index + direction
            if 0 <= new_index < len(self.queue):
                self.queue[index], self.queue[new_index] = self.queue[new_index], self.queue[index]

        self.checklist.rebuild(self.queue)

    def remove_current_image(self):
        if self.running:
            messagebox.showinfo("Experiment running", "Pause or abort the experiment before editing the queue.")
            return

        image_id = self.checklist.selected_item
        if image_id is None:
            messagebox.showinfo("No image selected", "Click an image in the list first.")
            return

        self.queue = [item for item in self.queue if item.image_id != image_id]
        self.clear_preview()
        self.checklist.rebuild(self.queue)
        self.raw_name_var.set("")
        self.status_var.set("Removed.")
        self.log(f"Removed {image_id} from the queue.")


    def parse_ground_truth_lines(self):
        raw_lines = self.gt_box.get("1.0", tk.END).strip().splitlines()
        parsed = []
        self.gt_extra_field_lines = []

        for line_number, line in enumerate(raw_lines, start=1):
            if "\t" in line:
                fields = line.split("\t")
            elif "|" in line:
                fields = line.split("|")
            elif ";" in line:
                fields = line.split(";")
            else:
                fields = [line]

            fields = [normalize(field) for field in fields]

            if any(fields[2:]):
                self.gt_extra_field_lines.append(line_number)

            while len(fields) < 2:
                fields.append("")

            parsed.append(tuple(fields[:2]))

        return parsed

    def apply_ground_truth(self):
        lines = self.parse_ground_truth_lines()

        for item in self.queue:
            if 0 <= item.load_index < len(lines):
                item.ground_truth = lines[item.load_index]


    def make_experiment_signature(self):
        return {
            "images": [
                {"path": str(Path(item.path).resolve()), "load_index": item.load_index,
                 "ground_truth": list(item.ground_truth)}
                for item in self.queue
            ],
            "engines": ENGINE_NAMES,
            "combinations": [list(combo) for combo in COMBINATIONS],
        }

    def save_checkpoint(self):
        data = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_signature": self.make_experiment_signature(),
            "results": self.results,
            "raw_baselines": self.raw_baselines,
            "completed_keys": list(self.completed_keys),
            "completed_image_ids": sorted(self.completed_image_ids),
        }

        temporary_file = CHECKPOINT_FILE.with_suffix(".tmp")
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        temporary_file.replace(CHECKPOINT_FILE)

    def load_checkpoint(self):
        if not CHECKPOINT_FILE.exists():
            return False

        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        signature = data.get("experiment_signature", {})

        saved_paths = {entry["path"] for entry in signature.get("images", [])}
        current_paths = {str(Path(item.path).resolve()) for item in self.queue}

        if saved_paths != current_paths:
            messagebox.showwarning(
                "Checkpoint mismatch",
                "The selected images do not match the images stored in the checkpoint.",
            )
            return False

        self.results = data.get("results", [])
        self.raw_baselines = data.get("raw_baselines", {})
        self.completed_keys = set(data.get("completed_keys", []))
        self.completed_image_ids = set(data.get("completed_image_ids", []))

        self.overwritten_image_ids.update(result.get("image_id", "") for result in self.results)

        self.log(f"Checkpoint loaded: {len(self.results)} result rows, {len(self.completed_keys)} completed tasks.")
        return True

    def load_checkpoint_dialog(self):
        if not self.queue:
            messagebox.showwarning(
                "Select images",
                "Select the same images used in the checkpoint before loading it.",
            )
            return

        if self.load_checkpoint():
            messagebox.showinfo("Checkpoint loaded", "Checkpoint loaded. You may now run or resume the experiment.")

    def request_pause(self):
        self.stop_requested = True
        self.status_var.set("Pause requested. Finishing the current OCR task...")
        self.log("Pause requested by user.")

    def request_abort(self):
        self.stop_requested = True
        self.abort_requested = True
        self.status_var.set("Abort requested. Stopping after the current OCR call...")
        self.log("Abort requested by user.")


    @staticmethod
    def task_key(image_id, engine, preprocessing, strength, block, c_value):
        return "|".join(str(value) for value in (image_id, engine, preprocessing, strength, block, c_value))

    def add_comparison_columns(self, result):
        image_id = result["image_id"]
        engine = result["engine"]

        baseline_key = f"{image_id}|{engine}"
        baseline = self.raw_baselines.get(baseline_key, {})

        raw_pooled = baseline.get("pooled_cer")
        result["raw_pooled_cer"] = raw_pooled

        if result["preprocessing"] == "raw":
            result["pooled_cer_improvement"] = 0.0
            result["pooled_cer_improvement_percent"] = 0.0
        else:
            result["pooled_cer_improvement"] = improvement(raw_pooled, result["pooled_cer"])
            result["pooled_cer_improvement_percent"] = improvement_percent(raw_pooled, result["pooled_cer"])

        return result


    def validate_and_prepare_run(self):
        if self.running:
            messagebox.showinfo("Experiment running", "An experiment is already running.")
            return False

        usable = [item for item in self.queue if item.crops is not None]
        if not usable:
            messagebox.showwarning(
                "No aligned images",
                "No selected image could be aligned. Each image must show all 4 black corner squares.",
            )
            return False

        self.apply_ground_truth()
        if self.gt_extra_field_lines:
            messagebox.showwarning(
                "Ground truth format",
                "Ground truth now has two fields per line: NAME | LRN. The Section field was removed.\n\n"
                "These lines have extra fields: "
                + ", ".join(str(number) for number in self.gt_extra_field_lines),
            )
            return False

        for item in usable:
            name_truth = normalize(item.ground_truth[0])
            if restrict_to_enye(name_truth) != name_truth:
                self.log(
                    f"Warning: the ground-truth name for {item.image_id} has an accented letter other "
                    "than the enye. The Name readers only produce the enye, so that accent will count as an error."
                )

        if not any(any(normalize(field) for field in item.ground_truth) for item in usable):
            messagebox.showwarning(
                "No ground truth",
                "Enter ground truth for at least one image before starting.",
            )
            return False

        not_ready = [name for name in ENGINE_NAMES if not self.engine_ready(name)]
        if not_ready:
            proceed = messagebox.askyesno(
                "Some engines are not ready",
                "These engines are still loading or failed to load and will be skipped for now:\n\n"
                + "\n".join(not_ready)
                + "\n\nStart anyway with the remaining engines?",
            )
            if not proceed:
                return False

        self.stop_requested = False
        self.abort_requested = False

        if CHECKPOINT_FILE.exists():
            try:
                self.load_checkpoint()
            except Exception as error:
                messagebox.showwarning("Checkpoint not loaded", f"Could not load checkpoint:\n{error}")

        return True

    def start_experiment(self):
        if not self.validate_and_prepare_run():
            return

        self.running = True
        thread = threading.Thread(target=self.run_experiment, daemon=True)
        thread.start()

    def start_single_run(self):
        if not self.validate_and_prepare_run():
            return

        strength = float(self.custom_strength_var.get())
        block = int(self.custom_block_var.get())
        c_value = int(self.custom_c_var.get())

        self.log(f"Starting a single run with strength={strength}, block={block}, C={c_value}.")

        configurations = [("combo", strength, block, c_value)]

        self.running = True
        thread = threading.Thread(target=self.run_experiment, args=(configurations,), daemon=True)
        thread.start()

    def update_progress(self, message, value, maximum=None):
        if maximum is None:
            maximum = max(int(self.progress.cget("maximum")), 1)
        percent = int(round(100 * value / maximum)) if maximum else 0
        percent = max(0, min(100, percent))

        self.root.after(0, lambda: self.status_var.set(message))
        self.root.after(0, lambda: self.progress.configure(value=value))
        self.root.after(0, lambda: self.progress_percent_var.set(f"{percent}%"))

    def append_result(self, item, engine, preprocessing, strength, block, c_value, evaluation,
                      ocr_seconds=None, preprocess_seconds=None, stage_times=None,
                      processing_seconds=None):
        result = {
            "image_id": item.image_id,
            "engine": engine,
            "preprocessing": preprocessing,
            "strength": strength,
            "block": block,
            "C": c_value,
            "name_cer": evaluation["name_cer"],
            "lrn_cer": evaluation["lrn_cer"],
            "pooled_cer": evaluation["pooled_cer"],
            "ocr_seconds": None if ocr_seconds is None else round(ocr_seconds, 4),
            "preprocess_seconds": None if preprocess_seconds is None else round(preprocess_seconds, 4),
            "processing_seconds": None if processing_seconds is None else round(processing_seconds, 4),
            "status": evaluation["status"],
            "gt_name_norm": evaluation["gt_name_norm"],
            "gt_lrn_norm": evaluation["gt_lrn_norm"],
            "pred_name": evaluation["pred_name"],
            "pred_lrn": evaluation["pred_lrn"],
            "pred_name_norm": evaluation["pred_name_norm"],
            "pred_lrn_norm": evaluation["pred_lrn_norm"],
            "raw_prediction": evaluation["raw_prediction"],
            "name_model": NAME_MODEL_INFO.get(engine, ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        stage_times = stage_times or {}
        for column in STAGE_COLUMNS:
            value = stage_times.get(column)
            result[column] = None if value is None else round(value, 5)

        if preprocessing == "raw":
            baseline_key = f"{item.image_id}|{engine}"
            self.raw_baselines[baseline_key] = {"pooled_cer": result["pooled_cer"]}
            result["raw_pooled_cer"] = result["pooled_cer"]
            result["pooled_cer_improvement"] = 0.0
            result["pooled_cer_improvement_percent"] = 0.0
        else:
            result = self.add_comparison_columns(result)

        self.results.append(result)
        self.save_cumulative_result(result)

    def load_cumulative_csv(self):
        if not CUMULATIVE_RESULTS_CSV.exists():
            return

        with open(CUMULATIVE_RESULTS_CSV, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            self.cumulative_rows = list(reader)

    def save_cumulative_result(self, result):
        image_id = result.get("image_id", "")

        if image_id not in self.overwritten_image_ids:
            self.cumulative_rows = [
                row for row in self.cumulative_rows if row.get("image_id") != image_id
            ]
            self.overwritten_image_ids.add(image_id)
            self.cumulative_rows.append(result)
            self.write_cumulative_csv()
        else:
            self.cumulative_rows.append(result)
            self.append_cumulative_row(result)

    @staticmethod
    def csv_fieldnames():
        return [
            "image_id", "engine", "preprocessing", "strength", "block", "C",
            "name_cer", "lrn_cer", "pooled_cer",
            "raw_pooled_cer", "pooled_cer_improvement", "pooled_cer_improvement_percent",
            "ocr_seconds", "preprocess_seconds", "processing_seconds", *STAGE_COLUMNS,
            "status", "gt_name_norm", "gt_lrn_norm",
            "pred_name", "pred_lrn", "pred_name_norm", "pred_lrn_norm",
            "raw_prediction", "name_model", "timestamp",
        ]

    def write_cumulative_csv(self):
        temporary_file = CUMULATIVE_RESULTS_CSV.with_suffix(".csv.tmp")
        with open(temporary_file, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_fieldnames(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.cumulative_rows)

        temporary_file.replace(CUMULATIVE_RESULTS_CSV)

    def append_cumulative_row(self, result):
        file_is_new = not CUMULATIVE_RESULTS_CSV.exists()
        with open(CUMULATIVE_RESULTS_CSV, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_fieldnames(), extrasaction="ignore")
            if file_is_new:
                writer.writeheader()
            writer.writerow(result)

    def run_experiment(self, configurations=None):
        try:
            if configurations is None:
                configurations = [("raw", None, None, None)]
                configurations.extend(
                    ("combo", strength, block, c_value) for strength, block, c_value in COMBINATIONS
                )

            active_engines = [name for name in ENGINE_NAMES if self.engine_ready(name)]
            usable_items = [item for item in self.queue if item.crops is not None]
            total_tasks = len(usable_items) * len(configurations) * len(active_engines)
            completed_count = len(self.completed_keys)

            initial_percent = int(round(100 * completed_count / max(total_tasks, 1)))
            self.root.after(
                0,
                lambda: self.progress.configure(maximum=max(total_tasks, 1), value=completed_count),
            )
            self.root.after(0, lambda: self.progress_percent_var.set(f"{initial_percent}%"))

            last_ui_update_time = 0.0
            last_checkpoint_time = 0.0
            UI_UPDATE_INTERVAL = 0.15
            CHECKPOINT_SAVE_INTERVAL = 2.0

            images_with_alignment_time = {
                row.get("image_id") for row in self.results
                if row.get("stage_align_seconds") not in (None, "")
            }

            for config_name, strength, block, c_value in configurations:
                if config_name == "raw":
                    setting_label = "raw"
                    setting_row = {"preprocessing": "raw"}
                else:
                    setting_label = f"strength={strength}, block={block}, C={c_value}"
                    setting_row = {
                        "preprocessing": "fractional_adaptive",
                        "strength": strength, "block": block, "C": c_value,
                    }
                config_did_work = False

                for item in usable_items:
                    if config_name == "raw":
                        preprocessing = "raw"
                        strength_value = "NA"
                        block_value = "NA"
                        c_value_result = "NA"
                    else:
                        preprocessing = "fractional_adaptive"
                        strength_value = strength
                        block_value = block
                        c_value_result = c_value

                    pending_engines = [
                        engine_name
                        for engine_name in active_engines
                        if self.task_key(
                            item.image_id, engine_name, preprocessing,
                            strength_value, block_value, c_value_result,
                        ) not in self.completed_keys
                    ]

                    if not pending_engines:
                        continue

                    config_did_work = True
                    item.status = "processing"
                    self.root.after(0, lambda current=item: self.checklist.update_row_text(current))

                    if self._last_auto_previewed_id != item.image_id:
                        self._last_auto_previewed_id = item.image_id
                        self.root.after(0, lambda image_id=item.image_id: self.select_preview_image(image_id))

                    preprocess_stage_times = {}
                    preprocess_start = time.perf_counter()
                    if config_name == "raw":
                        processed = {
                            field: pipeline_raw(crop, preprocess_stage_times)
                            for field, crop in item.crops.items()
                        }
                    else:
                        processed = {
                            field: pipeline_frac_adaptive(
                                crop, strength=strength, block=block, c_value=c_value,
                                timings=preprocess_stage_times,
                            )
                            for field, crop in item.crops.items()
                        }
                    preprocess_seconds = time.perf_counter() - preprocess_start
                    item_preprocess_seconds = preprocess_seconds

                    shared_stage_times = {
                        f"stage_{stage}_seconds": seconds
                        for stage, seconds in preprocess_stage_times.items()
                    }
                    if item.image_id not in images_with_alignment_time:
                        shared_stage_times.update(
                            {key: value for key, value in item.align_times.items() if value is not None}
                        )
                        images_with_alignment_time.add(item.image_id)

                    any_success_for_item = False

                    for engine_name in pending_engines:
                        key = self.task_key(
                            item.image_id, engine_name, preprocessing,
                            strength_value, block_value, c_value_result,
                        )

                        engine_stage_times = {}
                        ocr_start = time.perf_counter()
                        prediction = read_fields(processed, engine_name, engine_stage_times)
                        ocr_seconds = time.perf_counter() - ocr_start

                        score_start = time.perf_counter()
                        evaluation = evaluate_prediction(item.ground_truth, prediction)
                        engine_stage_times["score"] = time.perf_counter() - score_start
                        any_success_for_item = any_success_for_item or evaluation["status"] == "success"

                        self.append_result(
                            item=item, engine=engine_name, preprocessing=preprocessing,
                            strength=strength_value, block=block_value, c_value=c_value_result,
                            evaluation=evaluation,
                            ocr_seconds=ocr_seconds, preprocess_seconds=preprocess_seconds,
                            processing_seconds=(
                                item_preprocess_seconds + ocr_seconds + engine_stage_times.get("score", 0.0)
                            ),
                            stage_times={
                                **shared_stage_times,
                                "stage_ocr_name_seconds": engine_stage_times.get("ocr_name"),
                                "stage_ocr_lrn_seconds": engine_stage_times.get("ocr_lrn"),
                                "stage_score_seconds": engine_stage_times.get("score"),
                            },
                        )
                        preprocess_seconds = None
                        shared_stage_times = {}

                        self.completed_keys.add(key)
                        completed_count += 1
                        is_last_task = completed_count >= total_tasks
                        now = time.monotonic()

                        if AUTOSAVE_EVERY_RESULT and (
                            is_last_task or now - last_checkpoint_time >= CHECKPOINT_SAVE_INTERVAL
                        ):
                            self.save_checkpoint()
                            last_checkpoint_time = now

                        if is_last_task or now - last_ui_update_time >= UI_UPDATE_INTERVAL:
                            last_ui_update_time = now

                            self.update_progress(
                                f"{engine_name} | {item.image_id} | {completed_count}/{total_tasks}",
                                completed_count, maximum=total_tasks,
                            )
                            self.root.after(
                                0,
                                lambda current=item, eng=engine_name, cer=evaluation["pooled_cer"],
                                       cond=setting_label, secs=ocr_seconds:
                                    self.log(f"{current.image_id} | {eng} | {cond} -> "
                                             f"pooled CER = {'N/A' if cer is None else f'{cer:.3f}'} "
                                             f"| {secs:.2f} s"),
                            )

                        if self.stop_requested:
                            item.status = "pending"
                            self.root.after(0, lambda current=item: self.checklist.update_row_text(current))
                            self.save_checkpoint()
                            self.running = False
                            stopped_because = "aborted" if self.abort_requested else "paused"
                            self.root.after(
                                0,
                                lambda reason=stopped_because:
                                    self.status_var.set(f"Run {reason}. Press Start to resume."),
                            )
                            self.root.after(0, lambda reason=stopped_because: self.log(f"Run {reason} by user."))
                            return

                    item.status = "done" if any_success_for_item else "ocr_failed"
                    self.root.after(0, lambda current=item: self.checklist.update_row_text(current))
                    self.root.after(0, self.refresh_best_observed)

                if config_did_work:
                    this_key = combo_key(setting_row)
                    this_total = combo_time_totals(
                        [row for row in self.results if combo_key(row) == this_key]
                    ).get(this_key)
                    if this_total is not None:
                        self.root.after(
                            0,
                            lambda label=setting_label, info=this_total: self.log(
                                f"Setting [{label}] finished: {info['images']} image(s), "
                                f"{format_duration(info['seconds'])} total scanning time."
                            ),
                        )

            self.completed_image_ids.update(item.image_id for item in usable_items)

            self.save_checkpoint()
            self.running = False

            self.root.after(
                0,
                lambda: self.status_var.set(f"Experiment finished. {len(self.results)} result rows saved."),
            )
            self.root.after(0, lambda: self.log("Experiment finished."))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Finished",
                    "Experiment completed successfully.\n\n"
                    f"Result rows: {len(self.results)}\n"
                    f"Checkpoint: {CHECKPOINT_FILE}\n\n"
                    "Use \"Save results (CSV/JSON)\" to write the completed scans to a file.",
                ),
            )

        except Exception as error:
            error_message = str(error)

            self.save_checkpoint()
            self.running = False

            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Experiment error",
                    "The experiment stopped, but the checkpoint was saved.\n\n"
                    f"Error: {error_message}",
                ),
            )
            self.root.after(0, lambda: self.status_var.set("Stopped. Press Start to resume from checkpoint."))
            self.root.after(0, lambda: self.log(f"Experiment stopped on an error: {error_message}"))


    def build_timing_table(self, parent):
        columns = ["stage", "scope", "shared"] + list(ENGINE_NAMES)
        headings = {"stage": "Pipeline stage", "scope": "Runs", "shared": "All engines"}
        for engine in ENGINE_NAMES:
            headings[engine] = engine

        ttk.Label(
            parent,
            text="Mean time per run of each stage, from every result on file. Alignment is not part of the "
                 "per-setting total because it happens once per image.",
        ).pack(side="bottom", anchor="w", pady=(4, 0))

        self.timing_tree = ttk.Treeview(parent, columns=columns, show="headings")
        for column in columns:
            self.timing_tree.heading(column, text=headings[column])
            width = {"stage": 330, "scope": 190, "shared": 110}.get(column, 110)
            self.timing_tree.column(column, width=width, anchor="w" if column in ("stage", "scope") else "center")

        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.timing_tree.yview)
        self.timing_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.timing_tree.pack(side="left", fill="both", expand=True)

    def refresh_timing_tab(self, rows):
        self.timing_tree.delete(*self.timing_tree.get_children())
        if not rows:
            return

        for label, scope, shared_value, per_engine in timing_table_rows(stage_timing_stats(rows)):
            if per_engine:
                values = [label, scope, "-"] + [format_stage_time(per_engine.get(engine)) for engine in ENGINE_NAMES]
            else:
                values = [label, scope, format_stage_time(shared_value)] + ["-"] * len(ENGINE_NAMES)
            self.timing_tree.insert("", "end", values=values)

    def refresh_results_tab(self):
        rows = self.cumulative_rows
        self.matrix_tree.delete(*self.matrix_tree.get_children())
        self.combo_report_rows = []
        self.refresh_timing_tab(rows)

        if not rows:
            self.set_summary_text("No results on file yet. Run the experiment first.")
            return

        aggregated = aggregate_results(rows)
        time_totals = combo_time_totals(rows)
        per_engine_best = best_combo_per_engine(aggregated)
        overall_best = best_combo_overall(aggregated)
        engine_stats = overall_engine_stats(rows)

        self.set_summary_text(
            self.format_summary(aggregated, per_engine_best, overall_best, engine_stats, time_totals)
        )

        def overall_average(key):
            values = [
                aggregated[key][engine]["pooled_mean"]
                for engine in ENGINE_NAMES
                if engine in aggregated[key]
            ]
            return statistics.mean(values) if values else float("inf")

        for key in sorted(aggregated, key=overall_average):
            per_engine = aggregated[key]
            label = combo_label(key)

            counts = [per_engine[engine]["count"] for engine in ENGINE_NAMES if engine in per_engine]
            n_images = max(counts) if counts else 0

            row_values = [label, n_images]
            report_row = {
                "setting": label, "strength": key[1], "block": key[2], "C": key[3],
                "images": n_images,
            }

            for engine in ENGINE_NAMES:
                stats = per_engine.get(engine)
                if stats is None:
                    row_values += ["N/A", "N/A"]
                    report_row[f"{engine} pooled CER"] = ""
                    report_row[f"{engine} OCR sec per image"] = ""
                else:
                    ocr_mean = stats["ocr_mean"]
                    row_values += [
                        f"{stats['pooled_mean']:.4f}",
                        "N/A" if ocr_mean is None else f"{ocr_mean:.2f}",
                    ]
                    report_row[f"{engine} pooled CER"] = round(stats["pooled_mean"], 6)
                    report_row[f"{engine} OCR sec per image"] = "" if ocr_mean is None else round(ocr_mean, 4)

            timing = time_totals.get(key)
            if timing is None:
                row_values += ["N/A", "N/A"]
                report_row["combo total seconds"] = ""
                report_row["combo total time"] = ""
                report_row["seconds per image (all engines)"] = ""
            else:
                per_image = timing["seconds"] / timing["images"] if timing["images"] else None
                row_values += [
                    format_duration(timing["seconds"]),
                    "N/A" if per_image is None else f"{per_image:.2f} s",
                ]
                report_row["combo total seconds"] = round(timing["seconds"], 3)
                report_row["combo total time"] = format_duration(timing["seconds"])
                report_row["seconds per image (all engines)"] = "" if per_image is None else round(per_image, 4)

            self.matrix_tree.insert("", "end", values=row_values)
            self.combo_report_rows.append(report_row)

    def format_summary(self, aggregated, per_engine_best, overall_best, engine_stats, time_totals):
        lines = []

        lines.append("Best preprocessing setting per engine (lowest pooled CER):")
        for engine in ENGINE_NAMES:
            if engine in per_engine_best:
                key, value = per_engine_best[engine]
                ocr_mean = aggregated[key][engine]["ocr_mean"]
                time_text = "" if ocr_mean is None else f", OCR time {ocr_mean:.2f} s/image"
                lines.append(f"  {engine}: {combo_label(key)}  (pooled CER {value:.4f}{time_text})")
            else:
                lines.append(f"  {engine}: no results yet")

        if overall_best:
            key, value = overall_best
            lines.append("")
            lines.append(f"Best setting overall (averaged across engines): {combo_label(key)} "
                         f"(average pooled CER {value:.4f})")

        lines.append("")
        lines.append("Scanning time per setting (all images, all active engines):")
        timed = [
            (key, info["seconds"], info["images"], info["seconds"] / info["images"])
            for key, info in time_totals.items()
            if info["images"]
        ]
        if timed:
            fastest = min(timed, key=lambda entry: entry[3])
            slowest = max(timed, key=lambda entry: entry[3])
            lines.append(
                f"  Fastest: {combo_label(fastest[0])}  ({format_duration(fastest[1])} total, "
                f"{fastest[3]:.2f} s/image, {fastest[2]} image(s))"
            )
            lines.append(
                f"  Slowest: {combo_label(slowest[0])}  ({format_duration(slowest[1])} total, "
                f"{slowest[3]:.2f} s/image, {slowest[2]} image(s))"
            )
            lines.append(
                f"  Average per setting: {format_duration(statistics.mean(entry[1] for entry in timed))}"
                f"   |   All {len(timed)} setting(s) combined: "
                f"{format_duration(sum(entry[1] for entry in timed))}"
            )
        else:
            lines.append("  No timing on file yet.")

        lines.append("")
        lines.append("Overall statistics per engine, across every preprocessing setting:")
        for engine in ENGINE_NAMES:
            stats = engine_stats.get(engine, {})
            if stats.get("n", 0) == 0:
                lines.append(f"  {engine}: no results yet")
                continue
            mean = stats["mean"]
            median = stats["median"]
            stdev = stats["stdev"]
            stdev_text = f"{stdev:.4f}" if stdev is not None else "N/A"
            ocr_text = (
                "N/A" if stats["mean_ocr_seconds"] is None else f"{stats['mean_ocr_seconds']:.2f} s"
            )
            lines.append(
                f"  {engine}: n={stats['n']}, mean pooled CER={mean:.4f}, median={median:.4f}, "
                f"stdev={stdev_text}, mean OCR time/image={ocr_text}, successful={stats['success']}, "
                f"empty={stats['empty_prediction']}, invalid ground truth={stats['invalid_ground_truth']}"
            )

        return "\n".join(lines)

    def export_combo_report(self):
        self.refresh_results_tab()

        if not self.combo_report_rows:
            messagebox.showinfo("No results", "There are no results on file to export yet.")
            return

        path = filedialog.asksaveasfilename(
            title="Save combo report as CSV", defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        fieldnames = ["setting", "strength", "block", "C", "images"]
        for engine in ENGINE_NAMES:
            fieldnames += [f"{engine} pooled CER", f"{engine} OCR sec per image"]
        fieldnames += ["combo total seconds", "combo total time", "seconds per image (all engines)"]

        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.combo_report_rows)

        messagebox.showinfo("Saved", f"Combo report saved to:\n{path}")

    def set_summary_text(self, text):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")


    def save_results_file(self):
        all_rows = list(self.results)
        completed_rows = [dict(row) for row in all_rows if row.get("image_id") in self.completed_image_ids]
        unfinished_ids = {row.get("image_id") for row in all_rows} - self.completed_image_ids

        if not completed_rows:
            messagebox.showinfo(
                "No completed scans",
                "There are no completed scans to save yet.\n\n"
                "An image counts as completed once its run has finished, "
                "not while it is still running or after it was paused or stopped.",
            )
            return

        dialog_options = {
            "title": "Save completed scans (pick an existing file to add to it)",
            "defaultextension": ".csv",
            "filetypes": [("CSV or JSON", "*.csv *.json"), ("CSV files", "*.csv"), ("JSON files", "*.json")],
            "confirmoverwrite": False,
        }
        if self.last_results_path:
            dialog_options["initialdir"] = str(Path(self.last_results_path).parent)
            dialog_options["initialfile"] = Path(self.last_results_path).name

        try:
            path = filedialog.asksaveasfilename(**dialog_options)
        except tk.TclError:
            dialog_options.pop("confirmoverwrite")
            path = filedialog.asksaveasfilename(**dialog_options)

        if not path:
            return

        try:
            stats = save_results_merged(path, completed_rows, self.csv_fieldnames())
        except (ValueError, OSError) as error:
            messagebox.showerror("Could not save", str(error))
            return

        self.last_results_path = path
        message = (
            f"Saved to:\n{path}\n\n"
            f"New images added: {stats['added']}\n"
            f"Replaced (same file name): {stats['replaced']}\n"
            f"Other images kept as they were: {stats['kept']}\n"
            f"Rows in the file now: {stats['total_rows']}"
        )
        if unfinished_ids:
            message += f"\n\n{len(unfinished_ids)} image(s) are not finished yet and were not saved."

        self.log(
            f"Saved completed scans to {path} "
            f"(added {stats['added']}, replaced {stats['replaced']}, kept {stats['kept']})."
        )
        messagebox.showinfo("Saved", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRExperimentApp(root)
    root.mainloop()
