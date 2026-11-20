# Phase 6 — System Testing Report

## 1. Test objective

The Phase 6 test programme verifies that the full analytics and application chain
works with the real project artifacts:

```text
raw dataset -> feature engineering -> saved preprocessing pipeline ->
saved Random Forest -> prediction -> dashboard / marketing insights
```

The tests use the original raw dataset, the real Phase 4 selected model, the real
Phase 2 preprocessing pipeline, and the real Phase 3 K-Means pipeline. No
fabricated performance numbers, synthetic model results, or substitute models are
used.

## 2. Audit result

### 2.1 Expected artifacts

| Phase | Artifact | Status |
|---|---|---|
| Phase 1 | `reports/phase1/phase1_report.md`, EDA charts, summaries | Present |
| Phase 2 | `data/processed/preprocessing_summary.md`, feature dictionary, processed matrices, preprocessor | Present |
| Phase 3 | Baseline model reports, confusion matrices, K-Means artifacts, clustering metrics | Present |
| Phase 4 | Optimized model results, threshold analysis, model selection report, selected model artifact | Present |
| Phase 5 | Streamlit application and Phase 5 documentation | Present |
| Phase 6 | Tests, deployment files, and this report | Present after Phase 6 |

No essential Phase 1–5 artifact is missing. Phase 2's narrative report is stored in
`data/processed/preprocessing_summary.md` rather than a separate `reports/phase2`
folder; this is the completed Phase 2 output and not a missing component.

### 2.2 Artifact consistency checks

| Check | Result |
|---|---|
| Selected model | Random Forest |
| Decision threshold | 0.20 in artifact and metadata |
| Model feature count | 59 |
| Feature names in artifact vs metadata | Exact match |
| Model `n_features_in_` | 59 |
| `duration` in model features | Absent |
| Pipeline output shape on real raw sample | 50 × 59 |
| Saved app pipeline vs exported Phase 2 pipeline | Identical output |
| Processed test matrix shape | 9,043 × 59 |
| Test target positives | 1,058 |
| K-Means stages | `preprocessor`, `kmeans` |
| K-Means clusters | 4 |
| K-Means feature leakage (`y`, `duration`) | None |
| Test metrics recomputed from saved model | Exact match to Phase 4 metadata |

The test-set metrics recomputed independently from the saved model are:

| Metric | Value |
|---|---:|
| Accuracy | 0.8705075749 |
| Precision (yes) | 0.4543987086 |
| Recall (yes) | 0.5321361059 |
| F1 (yes) | 0.4902046147 |
| ROC-AUC | 0.8046476558 |
| PR-AUC | 0.4557211022 |
| Predicted positives | 1,239 |

These are actual results from `data/processed/X_test.csv` and
`data/processed/y_test.csv`; they were not invented for Phase 6.

## 3. Test environment

| Item | Value |
|---|---|
| Platform | Windows / PowerShell |
| Python | 3.14.6 |
| pytest | 9.1.1 |
| Streamlit | 1.62.0 |
| Pandas | 2.3.3 |
| NumPy | 2.3.3 |
| scikit-learn | 1.7.2 |
| Plotly | 6.9.0 |
| Joblib | 1.5.2 |

## 4. Automated pytest suite

### 4.1 Command

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pytest -q
```

### 4.2 Final result

```text
57 passed in 9.05s
```

### 4.3 Test distribution

| Test file | Tests | Coverage |
|---|---:|---|
| `tests/test_app_utils_and_filters.py` | 8 | Summary metrics, subscription rates, filters, formatting, metadata |
| `tests/test_data_and_features.py` | 7 | Dataset loading, values, unknown categories, feature engineering parity |
| `tests/test_model_and_validation.py` | 20 | Artifact contract, pipeline contract, test metrics, prediction outputs, validation |
| `tests/test_security_integrity.py` | 5 | Raw checksum, secrets scan, upload limit, destructive-call scan, ignore rules |
| `tests/test_segmentation.py` | 6 | K-Means contract, cluster counts, profiles, projection, predictions |
| `tests/test_streamlit_pages.py` | 11 | All pages, filters, prediction form, batch valid/invalid UI |

### 4.4 Required test areas

| Required area | Test evidence | Result |
|---|---|---|
| 1. Dataset loading | Raw shape/schema and app summary tests | PASS |
| 2. Missing and unknown value handling | Missing-value assertions, unknown category warning/prediction test | PASS |
| 3. Feature engineering | Parity with Phase 2, sentinel/bands, binary encoding | PASS |
| 4. Model loading | Artifact structure, feature contract, reloaded test metrics | PASS |
| 5. Individual prediction | Output format, probability range, threshold decision | PASS |
| 6. Batch prediction | Format, probabilities, predictions, no-duration path | PASS |
| 7. Invalid CSV input | Missing columns and non-numeric CSV through the Streamlit uploader | PASS |
| 8. Missing required columns | Utility-level and UI-level validation | PASS |
| 9. Incorrect data types | Numeric validation and UI error path | PASS |
| 10. K-Means loading/prediction | Pipeline contract, assignments, profiles, live prediction | PASS |
| 11. Prediction output format | Individual and batch output schemas | PASS |
| 12. Dashboard utility functions | Summary, rates, filters, formatting, metadata | PASS |

## 5. Edge cases tested

- Empty customer input.
- Missing required columns.
- Non-numeric age.
- Invalid binary values (`maybe`).
- Blank categorical values.
- Out-of-range age, day, campaign, previous, and pdays.
- Unknown category not seen in training (`astronaut`).
- Binary input supplied as 1/0.
- Prediction input without `duration`.
- Batch file with no required schema.
- Batch CSV with an incorrect type.
- Empty dashboard filter result.
- Sample-size filtering on grouping dimensions.
- Pandas/NaN handling in validation.
- Raw file checksum and destructive-write scan.

## 6. Streamlit application tests

All pages were rendered through Streamlit's `AppTest` harness:

| Page | Render | Interaction |
|---|---|---|
| Home | PASS | Artifact/model status renders |
| Executive Dashboard | PASS | Job filter applied without exception |
| Customer Prediction | PASS | Form submitted; probability and threshold `0.20` rendered; no duration field |
| Batch Prediction | PASS | 5-row CSV uploaded; 5 × 22 output preview; download button present |
| Customer Segmentation | PASS | K-Means and profile page renders |
| Marketing Analytics | PASS | Analytics page renders with filters |

Batch validation tests through the uploader:

- Valid raw CSV: scored successfully, download button present.
- Missing required columns: clear error and no download button.
- Non-numeric age: clear type error and no download button.

## 7. Live server startup

Command used:

```powershell
python -m streamlit run app/Home.py --server.headless true --server.port 8505
```

Actual response:

```text
HEALTH_STATUS=200
HEALTH_BODY=ok
PAGE_STATUS=200
PAGE_HAS_STREAMLIT=True
Uvicorn server started on :::8505
SERVER_STOPPED
```

The application started successfully with the real artifacts and the local
`.streamlit/config.toml` upload limit.

### Deployment checks

| Check | Command | Result |
|---|---|---|
| Python compilation | `python -m py_compile ...` across app, src, and tests | PASS |
| Dependency consistency | `python -m pip check` | No broken requirements found |
| Pytest suite | `python -m pytest -q` | 57 passed in 9.05s |
| Live health endpoint | `Invoke-WebRequest http://localhost:8505/_stcore/health` | HTTP 200 / `ok` |
| Live root page | `Invoke-WebRequest http://localhost:8505/` | HTTP 200 |

## 8. Existing Phase 5 smoke checks

Command:

```powershell
python src\phase5_app_checks.py
```

Actual result: all checks passed, including all six pages, feature parity, model
reload, individual/batch prediction, segmentation, and upload/download UI.

The harness emits benign Streamlit cache/context warnings when run outside a live
server. They are not application failures.

## 9. Security and data-quality audit

| Check | Result |
|---|---|
| Hardcoded API keys/passwords/paid AI clients | None found in application/source Python files |
| Environment variables required | No |
| `.env.example` required | No; intentionally not created |
| `.gitignore` covers `.env`, `.streamlit/secrets.toml`, caches, virtual environments | PASS |
| Upload type restriction | CSV only |
| Upload size limit | 50 MB in `.streamlit/config.toml` |
| Uploaded files persisted by app | No |
| Raw dataset checksum | PASS (`d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`) |
| Raw dataset write calls from app source | None detected |
| External APIs | None used |

## 10. Failure found and resolved during Phase 6

One initial pytest assertion assumed that at least one job group had 10,000
customers. That assumption was wrong for this dataset, so the test was corrected
to use a real threshold of 5,000 customers. This was a test-design issue, not an
application or model defect. After correction, all 57 tests passed.

## 11. Remaining issues and unautomated checks

- No functional test failures remain.
- Visual appearance was reviewed through code and page rendering, but no
  pixel-level screenshot regression suite exists.
- No multi-user concurrency or long-running memory/load test was performed.
- Very large CSV uploads near the 50 MB limit were not stress-tested.
- Browser-level accessibility automation was not run.
- The application is not a production deployment: authentication, TLS, external
  storage, monitoring, and formal privacy review would be required for that.
- The documented Phase 2/Phase 4 methodological limitation remains: preprocessing
  was fitted before CV. Phase 6 did not retrain or modify earlier models.

## 12. Conclusion

The complete local chain is operational: the real raw dataset loads, feature
engineering matches Phase 2, the saved preprocessing pipeline produces the expected
59 columns, the saved Random Forest reproduces the Phase 4 test metrics, the app
serves predictions and downloaded batch output, and the saved K-Means pipeline
reproduces the Phase 3 cluster assignments. The application starts successfully.
