# Phase 6 — Final Project Summary

## 1. Project objective

The project builds a Bank Marketing Intelligence System on the UCI Bank Marketing
dataset. It predicts term-deposit subscription likelihood, describes customer
segments, and provides campaign analytics in a local Streamlit application.

The application is intended for campaign prioritisation and exploratory decision
support. It does not establish causal effects and does not guarantee customer
behaviour.

## 2. Phase completion status

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Dataset exploration and data-quality analysis | Complete |
| Phase 2 | Cleaning, feature engineering, preprocessing, split | Complete |
| Phase 3 | Decision Tree, Random Forest, K-Means | Complete |
| Phase 4 | Evaluation, optimization, thresholding, final model selection | Complete |
| Phase 5 | Streamlit application | Complete |
| Phase 6 | Integration, pytest suite, security review, documentation, local deployment prep | Complete |
| Phase 7 | Not started | Not started |

No essential earlier-phase artifact is missing. The Phase 2 narrative report is
stored at `data/processed/preprocessing_summary.md` rather than in a separate
`reports/phase2` directory.

## 3. Dataset

- File: `data/raw/bank-full.csv`
- Rows: 45,211
- Columns: 17
- Target: `y` (`yes`/`no`)
- Positive class: 5,289 customers (`11.70%`)
- Negative class: 39,922 customers (`88.30%`)
- Missing cells: 0
- Duplicate rows: 0
- Raw checksum:
  `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`

Key Phase 1 findings carried into the model:

- `unknown` is a meaningful category and was retained.
- `pdays = -1` is a sentinel for no previous contact.
- Numeric outliers were retained rather than deleted.
- `duration` is a post-call variable and is excluded from pre-call prediction.

## 4. Preprocessing and feature engineering

Phase 2 produced:

- `age_group`
- `has_previous_contact`
- `contact_frequency_group`
- `previous_campaign_success`
- `pdays_since_previous_contact`
- Binary mapping for `default`, `housing`, and `loan`

The saved preprocessing pipeline expects 19 engineered inputs and produces 59
model columns:

- 6 scaled numeric columns.
- 5 passthrough binary columns.
- 48 one-hot encoded nominal columns.

Split:

- Training: 36,168 × 59; 4,231 positives.
- Test: 9,043 × 59; 1,058 positives.

Artifacts:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/preprocessor.joblib`
- `models/preprocessing_pipeline.joblib`

## 5. Data-mining models

### 5.1 Classification

Phase 3 established baselines. Phase 4 optimized both classifiers with
`RandomizedSearchCV`, 5-fold stratified cross-validation, and
`average_precision` scoring on training data. Thresholds were selected from
out-of-fold predictions by maximum F1; the test set was not used for tuning.

| Model | OOF threshold | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.73 | 0.8756 | 0.4704 | 0.5028 | 0.4861 | 0.7795 | 0.4178 |
| Random Forest | 0.20 | 0.8705 | 0.4544 | 0.5321 | 0.4902 | 0.8046 | 0.4557 |

Selected model: **Random Forest**, threshold **0.20**.

Confusion matrix on the untouched test set:

| Outcome | Count |
|---|---:|
| True negatives | 7,309 |
| False positives | 676 |
| False negatives | 495 |
| True positives | 563 |
| Predicted positives | 1,239 |

The Random Forest had the strongest OOF F1 and average precision. The Decision
Tree remains the more interpretable alternative. Phase 4 improved the Phase 3
Random Forest only modestly: F1 moved from 0.4862 to 0.4902.

### 5.2 Clustering

K-Means used 12 engineered customer and campaign-history features, with no target
and no duration.

Selected `k=4`:

- Inertia: 150,512.02
- Training silhouette: 0.1674
- Cluster sizes: 11,833 / 890 / 5,696 / 17,749
- Observed subscription rates: 10.39% / 14.49% / 21.01% / 9.44%

The silhouette is modest, so the segments are descriptive and exploratory.

## 6. Application

The Streamlit app contains:

- Home landing/status page.
- Executive Dashboard with filters and campaign charts.
- Customer Prediction form using the real saved model and preprocessing pipeline.
- Batch Prediction with CSV validation, scoring, and download.
- Customer Segmentation with actual K-Means assignments and profiles.
- Marketing Analytics with observed rates and non-causal caveats.

The pre-call prediction path excludes `duration`.

## 7. Integration verification

Phase 6 independently verified:

- Raw dataset and processed matrices.
- Model metadata and artifact contract.
- Exact feature-name consistency.
- Saved preprocessing output: 59 columns.
- Saved app pipeline equals exported Phase 2 pipeline.
- Reloaded Random Forest reproduces Phase 4 test metrics exactly:
  - F1 `0.49020461471484544`
  - ROC-AUC `0.8046476557534035`
  - PR-AUC `0.45572110223378215`
  - 1,239 predicted positives.
- K-Means cluster assignments match Phase 3 counts.
- `duration` is absent from classifier features and K-Means inputs.
- No `y` in K-Means inputs.
- The app starts successfully and returns HTTP 200 on the health endpoint.

## 8. Testing

Command:

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pytest -q
```

Result: **57 passed**.

Coverage includes:

- Dataset loading.
- Missing and unknown category handling.
- Feature engineering parity.
- Model loading and pipeline consistency.
- Individual prediction.
- Batch prediction and output format.
- Invalid CSV, missing columns, and incorrect types.
- K-Means loading, assignment, profiles, and prediction.
- Dashboard utility functions and filters.
- All Streamlit pages.
- Prediction form, batch upload, and download.
- Raw-data checksum, secret scan, upload configuration, and destructive-write scan.

Additional smoke checks:

```powershell
python src\phase5_app_checks.py
```

Result: all Phase 5 checks passed.

Live server check:

```text
HEALTH_STATUS=200
HEALTH_BODY=ok
PAGE_STATUS=200
```

## 9. Deployment preparation

Created or updated:

- `requirements.txt` — runtime dependencies.
- `requirements-dev.txt` — pytest development dependency.
- `.gitignore` — caches, environments, `.env`, and Streamlit secrets.
- `.streamlit/config.toml` — 50 MB upload limit, usage statistics disabled.
- `README.md` — overview, quick start, structure, testing, and links.
- `pytest.ini` — test discovery and reporting.
- Five Phase 6 reports under `reports/phase6/`.

No `.env.example` was created because the application requires no environment
variables. No paid APIs or external AI services are used.

## 10. Security and data quality

- Upload restricted to CSV and limited to 50 MB.
- Required columns, numeric types, binary values, ranges, blanks, and empty files
  are validated before prediction.
- Unknown categories are surfaced as warnings; the saved encoder handles them.
- Uploaded data is session-held and not persisted by the app.
- No hardcoded credentials or paid AI client imports were found.
- Raw dataset checksum is enforced by a test.
- The app does not overwrite the raw dataset or saved models.

## 11. Limitations

- Probabilities are model estimates, not guarantees.
- Observed analytics are associations, not causal effects.
- No campaign cost/revenue data exist, so threshold 0.20 is F1-oriented rather
  than profit-optimal.
- The K-Means silhouette is modest.
- Phase 2 preprocessing was fitted before Phase 4 CV; this makes CV estimates
  slightly optimistic for preprocessing.
- The OOF threshold shares training folds and is a validation estimate.
- The app is a local deployment preparation, not a production cloud service.
- Concurrency, very large uploads, and formal accessibility testing were not
  performed.

## 12. Recommended future improvements

1. Fit all preprocessing inside each CV fold during retraining.
2. Add probability calibration and cost-sensitive threshold selection.
3. Add temporal validation if future time-stamped campaigns become available.
4. Add campaign cost, revenue, and call-capacity data for profit optimisation.
5. Add model monitoring for drift, score distribution, and campaign outcomes.
6. Add fairness and disparate-impact review for customer treatment.
7. Add production authentication, HTTPS, structured logging, and persistent
   storage if the app is deployed beyond local use.
8. Add screenshot/visual regression and browser accessibility automation.

## 13. Final status

The essential Phase 1–6 components are present. The application uses the real
saved artifacts, starts successfully, and passes the automated tests. No Phase 6
failure remains open.

Phase 7 was not started.
