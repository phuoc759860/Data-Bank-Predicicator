# Phase 5 — Testing Report

## 1. Test environment

| Item | Value |
|---|---|
| Platform | Windows (PowerShell) |
| Python | 3.14.6 |
| Streamlit | 1.62.0 |
| Pandas | 2.3.3 |
| NumPy | 2.3.3 |
| scikit-learn | 1.7.2 |
| Plotly | 6.9.0 |
| Joblib | 1.5.2 |

The tests used the real project artifacts:

- `data/raw/bank-full.csv`
- `models/best_model.joblib`
- `models/best_model_metadata.json`
- `models/kmeans.joblib`
- `reports/phase3/clustering_metrics.csv`
- `data/processed/X_test.csv` and `data/processed/y_test.csv`

No substitute model, synthetic customer data, or fabricated prediction result was
used. The original dataset was not modified.

## 2. Commands executed

### Syntax compilation

```powershell
python -m py_compile src\app_utils.py app\Home.py app\pages\1_Executive_Dashboard.py app\pages\2_Customer_Prediction.py app\pages\3_Batch_Prediction.py app\pages\4_Customer_Segmentation.py app\pages\5_Marketing_Analytics.py app\components\theme.py app\components\charts.py app\components\filters.py app\utils\formatting.py
```

Result: no output, exit code 0.

### Full application smoke checks

```powershell
python src\phase5_app_checks.py
```

Result:

```text
==============================================================================
PHASE 5 APPLICATION CHECKS
==============================================================================
PASS artifacts and metadata
PASS model reload and prediction (F1=0.4902, positives=1,239)
PASS Phase 2 feature-engineering parity; duration absent
PASS individual prediction (No, probability=0.1511)
PASS batch prediction (5 rows)
PASS invalid batch input handling
PASS segmentation ({0: 11833, 1: 890, 2: 5696, 3: 17749})
PASS page: Home
PASS page: Executive Dashboard
PASS page: Customer Prediction
PASS page: Batch Prediction
PASS page: Customer Segmentation
PASS page: Marketing Analytics
PASS individual prediction form submission
PASS batch upload and download UI
PASS batch invalid-CSV error UI
==============================================================================
ALL PHASE 5 CHECKS PASSED
==============================================================================
```

The Streamlit testing harness prints benign `missing ScriptRunContext` and
`No runtime found, using MemoryCacheStorageManager` warnings when it runs outside
a live server. They are test-harness notices, not application errors.

### Real server startup

```powershell
python -m streamlit run app/Home.py --server.headless true --server.port 8505 --browser.gatherUsageStats false
```

Health check:

```text
HEALTH_STATUS=200
HEALTH_BODY=ok
PAGE_STATUS=200
PAGE_HAS_STREAMLIT=True
```

The server started successfully on port 8505 and was stopped after the check.

## 3. Test coverage and results

| Test | Method | Result |
|---|---|---|
| Artifact presence and metadata | `check_required_artifacts()`; metadata assertions | PASS — selected model is `Random Forest`, threshold `0.20`, duration excluded |
| Model loading | `joblib` reload through `load_best_model_bundle()` | PASS — model, preprocessing pipeline, feature names, and threshold validated |
| Model reload metrics | Saved model scored on untouched Phase 2 test set | PASS — F1 `0.4902`, 1,239 predicted positives |
| Phase 2 feature parity | Compared app engineering with `phase2_preprocessing.engineer_features` on 100 real rows | PASS — values match; `duration` absent |
| Individual prediction | Real pre-call record through `predict_customer()` | PASS — probability `0.1511`, prediction `No` |
| Individual prediction UI | Submitted the default Streamlit prediction form through `AppTest` | PASS — no exception; probability displayed; threshold displayed as `0.20` |
| Batch prediction | Scored 5 real raw rows through `predict_batch()` | PASS — all probabilities in `[0, 1]`, required output columns present |
| Batch upload UI | Uploaded a 5-row semicolon-separated raw CSV | PASS — 5 rows scored, 22 output columns, download button present |
| Invalid upload: missing columns | Uploaded a CSV with only `age,job` | PASS — clear error shown; no download button |
| Invalid input: non-numeric age | Validation layer with `age = "not-a-number"` | PASS — `ValueError` raised before prediction |
| Segmentation | Loaded saved K-Means; reproduced Phase 2 training split | PASS — cluster counts `{0: 11,833, 1: 890, 2: 5,696, 3: 17,749}` |
| Segmentation feature safety | Checked K-Means input feature names | PASS — no `y`, no `duration` |
| Home page | `AppTest` render | PASS — no exceptions |
| Executive Dashboard | `AppTest` render | PASS — no exceptions |
| Customer Prediction page | `AppTest` render | PASS — no exceptions |
| Batch Prediction page | `AppTest` render | PASS — no exceptions |
| Customer Segmentation page | `AppTest` render | PASS — no exceptions |
| Marketing Analytics page | `AppTest` render | PASS — no exceptions |
| Startup | Live Streamlit server and `/_stcore/health` | PASS — HTTP 200 / `ok` |

## 4. Model and prediction validation details

- The saved model is a dictionary artifact, not a bare estimator.
- The app verifies the saved preprocessing pipeline is fitted and can transform
  engineered input.
- The 59-column model matrix is created with the exact saved feature names.
- The app rejects inputs that do not produce 59 model columns.
- The model reload check independently reproduces the Phase 4 test F1 and the
  expected positive count.
- The prediction form never sends `duration`.
- The batch path drops `duration` if present and never uses `y` for scoring.

## 5. Segmentation validation details

The training split is reproduced with the same stratified 80/20 split and seed
used in Phase 2. The saved K-Means pipeline assigns the following actual training
counts:

| Cluster | Customers | Observed subscription rate |
|---|---:|---:|
| 0 | 11,833 | 10.39% |
| 1 | 890 | 14.49% |
| 2 | 5,696 | 21.01% |
| 3 | 17,749 | 9.44% |

The exact cluster counts match the Phase 3 outputs. The PCA view is display-only;
it is not used to fit or change the clusters.

## 6. Not tested automatically

- Pixel-level visual review of every viewport size.
- Browser accessibility audit (for example, axe or Lighthouse).
- Multi-user concurrency and long-running session behaviour.
- Very large batch files near Streamlit's upload limit.
- Every possible categorical-value combination in the prediction form.
- Real campaign cost/revenue optimisation, because those data are not available.

## 7. Known issues and notes

- No functional issues were found in the tested paths.
- The app is designed to fail clearly if Phase 3/4 artifacts are missing; it does
  not create replacement artifacts.
- Probability estimates and observed group rates are not guarantees or causal
  claims. This is stated in the UI and documentation.
- The test harness warnings about `ScriptRunContext` and cache storage are
  expected when using Streamlit components outside a running server.
- Phase 5 did not retrain or replace the selected model and did not modify the
  original dataset or the Phase 1–4 reports.
