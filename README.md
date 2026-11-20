# Bank Marketing Intelligence System

A local, end-to-end data science project that predicts term-deposit subscription
likelihood and segments bank marketing customers. The system covers exploratory
analysis, preprocessing, model development, evaluation, a Streamlit application,
integration testing, and deployment preparation.

## Project status

Phases 1–6 are present in this repository. The application uses the real saved
Phase 4 Random Forest artifact and the real Phase 3 K-Means artifact. No models
are retrained or replaced during application startup.

- Selected classifier: **Random Forest**
- Decision threshold: **0.20**
- Test F1 (yes class): **0.4902**
- Test ROC-AUC: **0.8046**
- Segmentation: **K-Means, k=4**
- Raw dataset: **45,211 customers × 17 columns**
- Canonical raw file: `data/raw/bank-full.csv`

## Quick start

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pip install -r requirements.txt
python -m streamlit run app/Home.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`).

## Main application pages

- **Home** — system status, model summary, and navigation guidance.
- **Executive Dashboard** — customer counts, subscription distribution, age, job,
  education, campaign performance, and sidebar filters.
- **Customer Prediction** — pre-call individual scoring with probability and
  threshold-based decision.
- **Batch Prediction** — CSV upload, validation, scoring, and downloadable results.
- **Customer Segmentation** — actual K-Means segments, profiles, and PCA view.
- **Marketing Analytics** — observed subscription rates by customer and campaign
  history dimensions, with correlation-versus-causation warnings.

## Project structure

```text
app/                    Streamlit pages, charts, filters, theme
data/raw/               Original UCI dataset and documentation
data/processed/         Phase 2 transformed matrices and preprocessing summary
models/                 Phase 3/4 saved models and metadata
reports/phase1/         EDA report, charts, and tables
reports/phase3/         Baseline model and clustering reports
reports/phase4/         Optimized model evaluation and selection reports
reports/phase5/         Application documentation and testing report
reports/phase6/         System architecture, testing, manual, installation, summary
src/                    Phase 1–4 pipelines, shared app utilities, smoke checks
tests/                  Phase 6 pytest suite
```

## Model and preprocessing

The selected model artifact is `models/best_model.joblib`. It contains:

- The fitted Random Forest classifier.
- The saved Phase 2 preprocessing pipeline.
- The 59 model feature names.
- The Phase 4 decision threshold (`0.20`).

`duration` is intentionally excluded from the primary pre-call prediction path.
The application validates raw inputs, applies the same feature engineering as
Phase 2, runs the saved preprocessing pipeline, and then applies the selected
threshold.

## Testing

Install development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Run the pytest suite:

```powershell
python -m pytest -q
```

Phase 6 result: **57 tests passed**. The suite covers data loading, unknown-value
handling, feature engineering, model loading, individual and batch prediction,
invalid inputs, K-Means, dashboard utilities, all Streamlit pages, security
checks, and raw-dataset integrity.

Additional smoke checks:

```powershell
python src\phase5_app_checks.py
```

## Documentation

- [System architecture](reports/phase6/system_architecture.md)
- [System testing report](reports/phase6/system_testing_report.md)
- [User manual](reports/phase6/user_manual.md)
- [Installation guide](reports/phase6/installation_guide.md)
- [Final project summary](reports/phase6/final_project_summary.md)
- [Phase 5 application documentation](reports/phase5/application_documentation.md)

## Security and data handling

- No paid APIs, external AI services, or API keys are used.
- No `.env.example` is required because the application needs no environment
  variables or secrets.
- Uploaded files are processed in the Streamlit session and are not persisted by
  the app.
- Uploads are restricted to CSV files and limited to 50 MB by
  `.streamlit/config.toml`.
- The original raw dataset is not modified by the application; a checksum test
  enforces this.
- Do not commit real credentials or `.streamlit/secrets.toml`; both are covered by
  `.gitignore`.

## Known limitations

- Model probabilities are estimates, not guarantees.
- Analytics show historical associations, not causal effects.
- No campaign cost or revenue data are available, so the threshold is not
  profit-optimal.
- K-Means has a modest silhouette score and is best used for exploration.
- Phase 2 preprocessing was fitted before cross-validation; this limitation is
  documented in the Phase 4 report.
- This is a local deployment preparation, not a production cloud deployment.

Phase 7 was not started.
