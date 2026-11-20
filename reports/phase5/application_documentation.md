# Phase 5 — Bank Marketing Intelligence Application

## 1. Purpose

This application turns the completed Phase 1–4 analysis into a decision-support
tool for a bank marketing team. It uses the original UCI Bank Marketing dataset,
the selected Phase 4 subscription model, and the Phase 3 K-Means segmentation
artifact. No model was retrained or replaced for the application.

The app supports:

- An executive view of the customer base and campaign performance.
- Individual pre-call subscription-probability scoring.
- Batch scoring from an uploaded CSV, with validation and CSV export.
- Descriptive customer segmentation from the real K-Means artifact.
- Marketing analytics with explicit correlation-versus-causation caveats.

## 2. Real artifacts used

| Artifact | Path | Use in the app |
|---|---|---|
| Original dataset | `data/raw/bank-full.csv` | Dashboard, analytics, batch examples, segmentation statistics |
| Phase 4 selected model | `models/best_model.joblib` | Individual and batch subscription scoring |
| Phase 4 metadata | `models/best_model_metadata.json` | Model identity, threshold, test metrics, provenance |
| Phase 3 K-Means model | `models/kmeans.joblib` | Customer segment assignment |
| Phase 3 clustering metrics | `reports/phase3/clustering_metrics.csv` | Inertia and silhouette display |
| Phase 2 processed test set | `data/processed/X_test.csv`, `data/processed/y_test.csv` | Independent reload/metric check only; not used for tuning |

The saved Phase 4 artifact is a dictionary containing:

- `model` — the selected Random Forest classifier.
- `preprocessing_pipeline` — the fitted Phase 2 imputer/scaler/one-hot pipeline.
- `feature_names` — the 59 model columns produced by that pipeline.
- `threshold` — the selected operating threshold, `0.20`.
- `model_name` — `Random Forest`.

The app loads and validates these components at runtime and stops with a clear
error if they are missing or malformed. It does not create a substitute model.

## 3. Leakage controls

- `duration` is **not** on the individual prediction form.
- `duration` is **not required** in batch uploads and is dropped if supplied.
- The feature-engineering layer removes `duration` before the saved
  preprocessing pipeline is applied.
- The K-Means input list contains no `y` and no `duration`.
- The target `y` is used only for observed descriptions in dashboards and for
  post-hoc cluster profile interpretation. It is never a model input.
- The test set is loaded only for a smoke check of the already-selected model;
  it is not used to tune parameters or thresholds in the app.

## 4. Application architecture

```text
app/
    Home.py
    pages/
        1_Executive_Dashboard.py
        2_Customer_Prediction.py
        3_Batch_Prediction.py
        4_Customer_Segmentation.py
        5_Marketing_Analytics.py
    components/
        charts.py
        filters.py
        theme.py
    utils/
        formatting.py

src/
    app_utils.py
    phase5_app_checks.py
```

### Responsibilities

| Layer | Responsibility |
|---|---|
| `src/app_utils.py` | Artifact loading and validation, data loading, feature engineering, batch prediction, segmentation assignment, cluster statistics |
| `app/components/theme.py` | Page configuration, CSS design system, KPI cards, insight callouts, Plotly theme, footer |
| `app/components/charts.py` | Reusable Plotly chart builders |
| `app/components/filters.py` | Sidebar filters and filtering logic |
| `app/pages/*` | Page-specific layout and interaction |
| `src/phase5_app_checks.py` | Reproducible smoke checks for artifacts, prediction, validation, segmentation, and all pages |

## 5. Pages

### Home

Landing page with actual dataset size, observed subscriber count and rate, the
selected model's test F1, artifact status, and a short guide to the application.
It also explains how to interpret model probabilities.

### 1. Executive Dashboard

Shows actual filtered customer counts, subscriber counts, non-subscribers, and
observed subscription rate. Charts include:

- Yes/No subscription distribution.
- Age distribution split by observed outcome.
- Job and education customer distributions.
- Subscription rate by contact channel.
- Subscription rate by previous campaign outcome.
- Subscription rate by contact month.
- Subscription rate by current-campaign contact frequency.

Sidebar filters cover job, education, marital status, contact channel, and age.

### 2. Customer Prediction

A pre-call form covering age, job, marital status, education, default status,
balance, housing loan, personal loan, contact channel, contact day and month,
current-campaign contacts, prior contacts, days since prior contact, and prior
campaign outcome.

The result displays:

- Predicted subscription probability.
- Selected decision threshold.
- An operational yes/no decision.
- A plain-language warning that the probability is an estimate, not a guarantee.

The default example is not a fixed or fabricated result: it is scored live by the
saved model.

### 3. Batch Prediction

Users upload a CSV containing the 15 required pre-call fields:

```text
age, job, marital, education, default, balance, housing, loan, contact,
day, month, campaign, pdays, previous, poutcome
```

Header names are case-insensitive. Numeric fields must be numeric. Binary fields
accept `yes`/`no` or `1`/`0`. `duration` and `y` are optional and ignored.

The page validates the file before scoring, reports missing columns or invalid
types, warns about categorical values not seen in training, and produces:

- Original input columns.
- `predicted_probability`.
- `predicted_subscription` (`Yes`/`No`).
- `predicted_label` (1/0).
- `decision_threshold`.
- `model_name`.

Results can be downloaded as `bank_marketing_predictions.csv`.

### 4. Customer Segmentation

The page reproduces the Phase 2 stratified 80/20 training split with seed 42,
applies the same feature engineering, and assigns clusters with the saved
K-Means pipeline. Displayed content includes:

- Cluster distribution and actual cluster sizes.
- A PCA projection for interactive visual exploration only.
- Actual training-split segment profiles.
- Mean age, median balance, top job, housing-loan share, previous-contact share,
  and observed subscription rate.
- Business interpretation for each segment.
- The exact 12 engineered K-Means input features.

K-Means uses no target label. The observed subscription rate is shown only as a
post-hoc description of each segment.

### 5. Marketing Analytics

Tabs compare observed subscription rates by job, age group, education, marital
status, previous campaign outcome, contact channel, month, and current-campaign
contact frequency. A minimum-group-size control prevents tiny groups from
dominating the charts. The page includes a group-summary table and explicit
warnings that observed associations are not causal effects.

## 6. Data and model flow

### Individual and batch prediction

```text
Raw pre-call fields
    -> validation
    -> Phase 2-compatible feature engineering
       (binary mapping, prior-contact flags, age/contact-frequency bands,
        pdays treatment, duration removal)
    -> saved Phase 2 preprocessing pipeline
       (imputation, scaling, one-hot encoding)
    -> saved Random Forest classifier
    -> probability
    -> threshold 0.20
    -> Yes/No operational decision
```

The application's feature-engineering function is validated against
`src/phase2_preprocessing.py:engineer_features` on real raw rows. The parity check
is part of `src/phase5_app_checks.py`.

### Segmentation

```text
Raw training rows
    -> same Phase 2-compatible feature engineering
    -> saved K-Means preprocessing pipeline
    -> saved K-Means model
    -> cluster label
    -> descriptive profile statistics
```

## 7. Selected model at a glance

The Phase 4 selected model is a Random Forest with operating threshold `0.20`.
The Phase 4 test-set metrics stored in the metadata are:

| Metric | Value |
|---|---:|
| Accuracy | 0.8705 |
| Precision (yes) | 0.4544 |
| Recall (yes) | 0.5321 |
| F1 (yes) | 0.4902 |
| ROC-AUC | 0.8046 |
| PR-AUC | 0.4557 |

The independent reload check in `src/phase5_app_checks.py` reproduces F1
`0.4902` and 1,239 predicted positives on the untouched test set.

## 8. Design notes

- Professional navy/teal/gold banking palette with a light workspace.
- Consistent typography, spacing, cards, and section hierarchy.
- Sidebar navigation is provided by Streamlit multipage routing.
- Plotly charts are interactive and responsive.
- Animations are disabled in Plotly layouts to keep the interface calm.
- Empty filter results and invalid uploads produce explicit messages rather than
  misleading charts or predictions.
- The app is read-only with respect to the original data, models, and reports.

## 9. Running the application

From the project root:

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pip install -r requirements.txt
python -m streamlit run app/Home.py
```

Then open the local URL printed by Streamlit (normally
`http://localhost:8501`).

Optional fixed port:

```powershell
python -m streamlit run app/Home.py --server.port 8501
```

Headless check:

```powershell
python -m streamlit run app/Home.py --server.headless true --server.port 8505
```

## 10. Re-running the automated checks

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python src\phase5_app_checks.py
```

This runs artifact validation, model reload, Phase 2 feature parity, individual
prediction, batch prediction, invalid-input checks, segmentation checks, all six
pages through Streamlit's `AppTest`, the prediction form submission, and valid
and invalid batch-upload UI tests.

## 11. Limitations and responsible use

- A model probability is an estimate from historical patterns, not a guarantee.
- Observed group rates are descriptive associations; they are not causal effects.
- The dataset does not contain campaign cost, revenue, or customer lifetime value,
  so the app cannot optimise profit or prescribe contact capacity.
- The Phase 4 threshold was selected from out-of-fold training predictions and is
  model-specific; it is not a universal business rule.
- Phase 2 preprocessing was fitted before Phase 4 cross-validation; this is a
  known methodological limitation carried forward from the completed phases.
- The segmentation silhouette is modest, so segments should guide exploration
  rather than serve as rigid customer definitions.
- Uploaded files are processed in the Streamlit session and are not persisted by
  the application. The app uses no paid APIs and no external AI services.
- Human review is required before using scores for customer treatment or any
  decision with material consequences.

## 12. Scope boundary

Phase 5 adds the web application, tests, and documentation only. It does not
retrain models, regenerate Phase 1–4 outputs, or start Phase 6.
