# CardioLens — AI + Health, Intermediate

An explainable heart disease classification research demo for the Veritas AI International High School Hackathon 2026, brief 4.1.2. Only the AI + Health intermediate project is included.

**This project is a prototype and is not medical advice.**

## User and workflow
A nurse exploring an already-completed adult cardiac checkup can load a public test case, edit 13 measurements, inspect the model score and its strongest contributions, and compare the prediction with held-out evidence. Several features require specialized clinical tests; this is not a home screening app. Inputs remain in browser memory and are not saved or transmitted. Use public or fictional cases only.

## Run the app
No API keys, server inference, or build step are required:

```bash
python -m http.server 8000
```
Open http://localhost:8000. Python serves files only; inference runs in JavaScript using the actual trained coefficients in model.json.

## Reproduce training
Python 3.11+ recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```
On Windows activate with `.venv\Scripts\activate`. The public processed Cleveland dataset is included so training does not depend on a network request. If absent, train.py downloads the original UCI archive.

Data: 303 Cleveland records. Target: num > 0. Stratified 80/20 split with seed 42: 242 train / 61 test. No hyperparameter search or test-set-based model selection. All imputation, standardization, and one-hot encoding fit on training data only. Numeric missing values use the training median, categorical missing values the training mode. Logistic regression: C=1, max_iter=2000. Numeric fields: age, trestbps, chol, thalach, oldpeak. Other fields are categorical. The exported coefficients implement the same transformation in the browser.

## Actual held-out results at threshold 0.50

| Metric | Logistic regression | Majority-class baseline |
| --- | ---: | ---: |
| Accuracy | 88.52% | 54.10% |
| Disease recall | 92.86% | 0.00% |
| Precision | 83.87% | 0.00% (no positive predictions) |
| True positives | 26 | 0 |
| True negatives | 28 | 33 |
| Missed cases (FN) | 2 | 28 |
| False alarms (FP) | 5 | 0 |

Model ROC AUC: 0.96645. These are descriptive results for one small historical test split, not clinical validation. No confidence interval is claimed.

## Our additions
- A nurse-oriented workflow with feature contribution explanations and explicit missing-value handling.
- An interactive threshold trade-off with separate false negatives and false positives.
- Sex and age subgroup reporting, including sample counts and recall.
- A real held-out failure case that can be loaded into the form. The exact cause of an individual error is unknown; overlapping patterns and limited data are plausible contributors.

## Limitations and honest failure analysis
The cohort is small, historical, and clinically selected. It does not represent a general population or people in Kazakhstan. Many features require clinical tests. Predictions classify the historical disease-present label, not future risk. The probabilities have not been independently calibrated. A low score does not rule out disease; this prototype cannot determine treatment, urgency, or diagnosis. Coefficient contributions are associations, not causes. Sex is encoded only as male/female in the source; this is a dataset limitation. Small subgroup counts make comparisons uncertain. No external-hospital evaluation or real-person usability study was performed. The UI threshold starts at 0.50, chosen before evaluation; exploring other thresholds is educational and must not be presented as independent test-set optimization.

## Submission guide
1. Show loading a public test case and explaining a prediction.
2. Compare the model with the majority-class baseline.
3. Show the 2 missed cases / 5 false alarms at 0.50, then explore the threshold slider.
4. Show subgroup sizes and a real model error.
5. Explain data limitations, the intended user, and what your team contributed.

Team members must understand the code, add their actual names/contributions, disclose coding assistance, and record their own demo. Do not claim user testing that did not occur.

## Sources and attribution
- Janosi, A., Steinbrunn, W., Pfister, M., & Detrano, R. (1988). *Heart Disease*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X . CC BY 4.0. Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease ; original processed.cleveland.data included unchanged. Missing markers are `?`.
- Project specification: Veritas AI International High School Hackathon 2026 Project Briefs, 4.1.2.
- scikit-learn: https://scikit-learn.org/stable/ ; pandas: https://pandas.pydata.org/ ; NumPy: https://numpy.org/ . Python for training; HTML/CSS/JavaScript for inference and UI.
- Google Fonts: DM Sans and Manrope, open-source font families delivered by Google Fonts. System sans-serif fallback works offline.
- Coding and documentation assistance: OpenAI ChatGPT. No generative medical API is used.

## Deployment
Import this repository into Vercel, choose Other framework, leave build and install commands empty, and deploy the repository root. vercel.json configures static hosting. Re-training is a deliberate local step; deployments serve the committed model.json.
