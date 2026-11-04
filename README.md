# Effect of Image Preprocessing on OCR Accuracy of Handwritten Student Identifiers

Code and data for the paper **"Effect of Image Preprocessing on OCR Accuracy of Handwritten Student Identifiers: A Comparison of Tesseract, EasyOCR and PaddleOCR"**.

The study measures the effect of a denoise, sharpen and adaptive-threshold pipeline (100 parameter combinations) on the recognition of the handwritten Name and 12-digit Learner Reference Number (LRN) fields of an answer sheet, using three open-source OCR engines: Tesseract, EasyOCR and PaddleOCR.

---

## 1. Files

| File | Description |
|---|---|
| `ocr_gui.py` | Experiment program (Tkinter GUI). Aligns each sheet, crops the Name and LRN fields, applies the preprocessing settings, runs the three engines and scores the results. |
| `Answer_Sheet_Name_LRN_A4.png` | A4 answer-sheet template. It must be in the same folder as `ocr_gui.py`. |
| `ocr_analysis.py` | Analysis program. Applies the selection rule and the paired bootstrap to the saved results. |
| `requirements.txt` | Python packages and versions. |
| `LICENSE` | MIT License. |
| `CITATION.cff` | Citation metadata (used by GitHub's "Cite this repository" button). |
| `.zenodo.json` | Metadata for the Zenodo archive of each release. |
| `ground_truth_and_conditions.csv` | Ground truth (name and LRN) and the paper or capture condition of each image. |
| `images/` | The 30 photographs, `image1.jpg` to `image30.jpg`. |

**Run the experiment**

```
python ocr_gui.py
```

Select the images, enter the ground truth (one line per image, in the same order: `NAME | LRN`), wait until the three engines show **Ready**, then click **Start**. Results are saved to `ocr_experiment_answersheet_all_results.csv`. Progress is saved to `ocr_experiment_answersheet_checkpoint.json`, and **Load Checkpoint** resumes an interrupted run.

**Run the analysis**

```
python ocr_analysis.py ocr_experiment_answersheet_all_results.csv -o analysis_output --expected-sheets 28
```

To add a table by condition, pass the conditions file with `--conditions ground_truth_and_conditions.csv`.

The outputs are written to `analysis_output`: `analysis_report.md`, `summary_table.csv`, `all_combos.csv`, `bootstrap.csv`, `parameter_effects.csv`, `top_combo_errors.csv` and `analysis_results.json`.

---

## 2. Data

- **Images:** `image1` to `image30` in the `images/` folder (30 photographs of answer sheets, each written by a different person).
- **Analysed:** 28 sheets. `image24` and `image25` were excluded because the four corner marks could not be detected, so they cannot be aligned.
- **Content:** all names and LRNs are fictitious and were chosen by the researcher. Each writer copied the assigned name and LRN, so the only personal element is their handwriting. All writers consented to the use of their handwriting. No real learner data are present.
- **Ground truth:** one line per image, in the order the images are loaded, in the form `NAME | LRN` (surname first, then first name; 12 digits after the `|`). The same ground truth, with the paper or capture condition of each image (for example crumpled, wet, torn, stained, dim lighting), is in `ground_truth_and_conditions.csv`.

**Capture setup**

| Item | Value |
|---|---|
| Camera | Android 13 smartphone, 108 MP triple-camera system, aperture f/1.75, 1/1.67-inch sensor |
| Distance | at most 10 inches (about 25 cm) |
| Lighting | normal room lighting, no flash |
| Image files | JPEG, 1536 x 2048 pixels, no manual cropping or editing |

---

## 3. Software versions

All engines ran on the CPU only (no GPU), on Windows 11.

| Component | Version |
|---|---|
| Python | 3.13 |
| Tesseract (program) | 5.4.0 (build 20240606, Leptonica 1.84.1) |
| pytesseract | 0.3.13 |
| EasyOCR | 1.7.2 |
| PaddleOCR | 3.7.0 (PP-OCRv6 medium models) |
| PaddlePaddle | 3.3.1 |
| PaddleX | 3.7.2 |
| PyTorch / torchvision | 2.14.0 / 0.29.0 |
| OpenCV | 4.10.0 |
| NumPy | 2.2.6 |
| Pillow | 10.4.0 |
| pandas | 2.2.3 |
| SciPy | 1.14.1 |

Install the Python packages with:

```
python -m pip install -r requirements.txt
```

Notes:

- Tesseract is a separate program, not a Python package. Install it, make sure `tesseract --version` works in a new Command Prompt (otherwise add `C:\Program Files\Tesseract-OCR` to `PATH`), and place its language data files in its `tessdata` folder. Tesseract's output depends on the exact language data files installed.
- Install only one OpenCV package (`opencv-contrib-python`). Do not install `opencv-python` or `opencv-python-headless` next to it, because they share one `cv2` folder and can change which version loads.
- The first run downloads the EasyOCR and PaddleOCR models, so it needs an internet connection. Later runs use the cache.

---

## 4. License and citation

This repository is released under the MIT License (see `LICENSE`). GitHub's **Cite this repository** button uses `CITATION.cff`. Each release is archived on Zenodo [add the Zenodo DOI after the first release].

If you use this code or data, please cite:

Perez, K. P., & Talirongan, F. J. B. ([year]). Effect of image preprocessing on OCR accuracy of handwritten student identifiers: A comparison of Tesseract, EasyOCR and PaddleOCR. [Venue].

```bibtex
@article{PerezTalirongan2026,
  title   = {Effect of Image Preprocessing on OCR Accuracy of Handwritten Student Identifiers:
             A Comparison of Tesseract, EasyOCR and PaddleOCR},
  author  = {Perez, Kenneth P. and Talirongan, Florence Jean B.},
  journal = {[Venue]},
  year    = {2026}
}
```

**Authors**

- **Kenneth P. Perez** (corresponding author), School of Graduate Studies, University of Northwestern Mindanao, Philippines. Email: markyheras@gmail.com. ORCID: [0009-0007-9546-0972](https://orcid.org/0009-0007-9546-0972)
- **Florence Jean B. Talirongan**, College of Computer Studies, Misamis University, Philippines. ORCID: [0000-0002-3775-8208](https://orcid.org/0000-0002-3775-8208)
