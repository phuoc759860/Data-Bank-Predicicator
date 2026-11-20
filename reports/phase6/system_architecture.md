# Phase 6 — System Architecture

## 1. Project objective

The Bank Marketing Intelligence System predicts whether a customer is likely to
subscribe to a term deposit and provides descriptive customer segments and
campaign analytics. The intended use is campaign prioritisation: rank customers
for outreach, understand broad customer groups, and inspect historical conversion
patterns without claiming causality.

The system is a local Streamlit application built on the original UCI Bank
Marketing dataset and the saved Phase 2–4 artifacts. It does not retrain models
when it starts.

## 2. Dataset and attributes

### 2.1 Source

- Canonical raw file: `data/raw/bank-full.csv`
- Format: semicolon-separated CSV
- Rows: 45,211
- Columns: 17
- Target: `y` (`yes` / `no`)
- Observed positives: 5,289 (`11.70%`)
- Observed negatives: 39,922 (`88.30%`)
- Missing cells: 0
- Duplicate rows: 0

The raw file is preserved read-only by the application. Additional files under
`data/raw/` are retained but are not the computational source for the application.

### 2.2 Attribute roles

| Attribute | Type | Role |
|---|---|---|
| `age` | numeric | Customer age |
| `job` | categorical | Occupation |
| `marital` | categorical | Marital status |
| `education` | categorical | Education level |
| `default` | binary categorical | Credit default status |
| `balance` | numeric | Average yearly balance |
| `housing` | binary categorical | Housing loan |
| `loan` | binary categorical | Personal loan |
| `contact` | categorical | Contact communication channel |
| `day` | numeric | Last contact day of month |
| `month` | categorical | Last contact month |
| `duration` | numeric | Last call duration; **excluded from pre-call prediction** |
| `campaign` | numeric | Contacts in current campaign |
| `pdays` | numeric | Days since previous contact; `-1` means no previous contact |
| `previous` | numeric | Contacts before this campaign |
| `poutcome` | categorical | Previous campaign outcome |
| `y` | binary target | Term-deposit subscription (`yes`/`no`) |

`duration` is not available before a call and would leak post-contact information.
The prediction path never requires it. It may remain in the raw file for
descriptive context but is dropped before model preprocessing.

### 2.3 Data-quality decisions inherited from Phase 1/2

- Missing cells were absent, but the string category `unknown` was retained and
  treated as information rather than silently deleted.
- The `pdays = -1` sentinel was separated into `has_previous_contact` and
  `pdays_since_previous_contact`.
- Numeric outliers were retained rather than deleted or winsorised.
- `duration` was excluded from model inputs.
- The Phase 2 split was stratified 80/20 with `random_state=42`:
  - Training: 36,168 rows; 4,231 positive (`11.70%`)
  - Test: 9,043 rows; 1,058 positive (`11.70%`)

Unknown-value detail is recorded in `data/processed/unknown_values_investigation.csv`
and in the Phase 2 summary.

## 3. Data preprocessing

Phase 2 produced the feature engineering and fitted preprocessing pipeline used
by later phases and the app.

### 3.1 Engineered features

| Feature | Source | Purpose |
|---|---|---|
| `age_group` | `age` | Captures non-linear age patterns |
| `has_previous_contact` | `pdays` | Removes the `-1` sentinel ambiguity |
| `contact_frequency_group` | `campaign` | Groups current-campaign contact intensity |
| `previous_campaign_success` | `poutcome` | Flags successful prior campaign outcomes |
| `pdays_since_previous_contact` | `pdays` | Recency without the `-1` sentinel |
| `default`, `housing`, `loan` | yes/no | Mapped to 0/1 binary columns |

The final engineered frame has 19 inputs used by the preprocessing pipeline:

```text
age, job, marital, education, default, balance, housing, loan, contact,
day, month, campaign, previous, poutcome, has_previous_contact,
previous_campaign_success, age_group, contact_frequency_group,
pdays_since_previous_contact
```

### 3.2 Transformation

The saved Phase 2 `Pipeline` uses a `ColumnTransformer`:

- Numeric: `StandardScaler` on `age`, `balance`, `day`, `campaign`, `previous`,
  and `pdays_since_previous_contact`.
- Binary: passthrough on `default`, `housing`, `loan`, `has_previous_contact`,
  and `previous_campaign_success`.
- Nominal: `OneHotEncoder(handle_unknown="ignore")` on `job`, `marital`,
  `education`, `contact`, `month`, `poutcome`, `age_group`, and
  `contact_frequency_group`.

Output: 59 model columns. `models/preprocessing_pipeline.joblib` and the
pipeline embedded in `models/best_model.joblib` transform features identically;
this is covered by a pytest check.

### 3.3 Leakage controls

- Transformations were fitted on the training split only in Phase 2.
- The test set was untouched during Phase 4 search, threshold selection, and
  model selection.
- The app does not refit preprocessing.
- The app never uses `duration` for scoring.
- K-Means uses no `y` and no `duration`.

Known methodological limitation: Phase 2 scaling/encoding was fitted before
Phase 4 cross-validation. This is retained as a documented limitation rather than
silently rerunning earlier phases. A production retraining workflow should fit
preprocessing inside every CV fold.

## 4. Data-mining algorithms

### 4.1 Classification

Phase 3 trained a Decision Tree and a Random Forest on the processed training
matrix and evaluated them on the untouched test set.

Phase 4 optimized both models with `RandomizedSearchCV`, using
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and
`average_precision` scoring on training data only. Thresholds were selected from
out-of-fold training predictions by maximum F1.

| Model | Selected parameters | OOF threshold | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|
| Decision Tree | `max_depth=10`, `min_samples_leaf=50`, `class_weight='balanced'` | 0.73 | 0.4861 | 0.7795 | 0.4178 |
| Random Forest | `n_estimators=400`, `max_depth=18`, `min_samples_leaf=10`, `min_samples_split=5`, `max_features=0.3` | 0.20 | 0.4902 | 0.8046 | 0.4557 |

Selected model: **Random Forest**, because it had the strongest out-of-fold F1 and
average precision. The Decision Tree remains the more interpretable alternative.

### 4.2 Clustering

Phase 3 fitted K-Means on 12 engineered customer-history features after
preprocessing:

```text
age, balance, previous, default, housing, loan, has_previous_contact,
previous_campaign_success, job, marital, education, poutcome
```

Selected `k=4`:

- Inertia: 150,512.02
- Training silhouette: 0.1674
- Training cluster sizes: 11,833 / 890 / 5,696 / 17,749
- Observed subscription rates by cluster: 10.39% / 14.49% / 21.01% / 9.44%

The silhouette is modest, so segments are descriptive exploration aids rather
than rigid business definitions.

## 5. System architecture

### 5.1 Logical layers

```text
Raw data and saved artifacts
        |
        v
Data access and artifact validation          src/app_utils.py
        |
        v
Feature engineering and preprocessing        src/app_utils.py + saved pipeline
        |
        v
Prediction / segmentation logic              saved RF + saved K-Means
        |
        v
Visualization and UI components              app/components/
        |
        v
Streamlit pages                              app/Home.py + app/pages/
```

### 5.2 Runtime flow: individual or batch prediction

```text
Raw pre-call fields
  -> validate_raw_input()
  -> build_engineered_features()
  -> saved preprocessing pipeline
  -> 59-column matrix with saved feature names
  -> Random Forest predict_proba()
  -> threshold 0.20
  -> probability + Yes/No decision
```

The app verifies that the pipeline output has exactly 59 columns and that the
feature names match the saved model contract.

### 5.3 Runtime flow: segmentation

```text
Raw training split (same stratified seed as Phase 2)
  -> same feature engineering
  -> saved K-Means preprocessing pipeline
  -> saved K-Means model
  -> cluster label
  -> actual profile statistics and descriptive observed rates
```

PCA is used only to display the clusters in two dimensions; K-Means does not use
the PCA projection.

### 5.4 Key modules

| Module | Responsibility |
|---|---|
| `src/app_utils.py` | Paths, artifact validation, data loading, validation, feature engineering, prediction, segmentation, profile statistics |
| `src/phase1_exploration.py` | Phase 1 EDA and reports |
| `src/phase2_preprocessing.py` | Split, feature engineering, fitting/saving preprocessing |
| `src/phase3_models.py` | Baseline classifiers and K-Means |
| `src/phase4_evaluation.py` | Optimization, threshold analysis, final selection |
| `src/phase5_app_checks.py` | Phase 5 smoke checks retained for compatibility |
| `app/components/theme.py` | Consistent banking dashboard design system |
| `app/components/charts.py` | Reusable Plotly chart builders |
| `app/components/filters.py` | Shared sidebar filters |
| `app/pages/*` | Streamlit page layouts and interactions |
| `tests/*` | Phase 6 pytest integration and security tests |

## 6. Artifact contract

| Artifact | Contract |
|---|---|
| `models/best_model.joblib` | Dictionary with `model`, `preprocessing_pipeline`, `feature_names`, `threshold`, `model_name` |
| `models/best_model_metadata.json` | Selected model, threshold, parameters, OOF/test metrics, feature lists |
| `models/preprocessing_pipeline.joblib` | Fitted Phase 2 preprocessing pipeline |
| `models/kmeans.joblib` | Pipeline with `preprocessor` and `kmeans` steps |
| `data/processed/X_train.csv`, `X_test.csv` | 59-column transformed matrices |
| `data/processed/y_train.csv`, `y_test.csv` | Binary targets |
| `reports/phase3/clustering_metrics.csv` | k-selection evidence |
| `reports/phase4/final_model_selection.md` | Selection rationale and limitations |

The app stops with an explicit error if a required artifact is missing. It never
creates a substitute model.

## 7. Application functionality

- **Home:** artifact status and model summary.
- **Executive Dashboard:** observed customer and campaign distributions with
  filters.
- **Customer Prediction:** pre-call individual scoring and probability display.
- **Batch Prediction:** validated CSV upload, scoring, and downloadable output.
- **Customer Segmentation:** actual K-Means assignments, profiles, and PCA view.
- **Marketing Analytics:** observed rates by customer and campaign dimensions,
  with explicit non-causal interpretation.

## 8. Deployment architecture

This is a local deployment preparation:

- Streamlit server launched with `python -m streamlit run app/Home.py`.
- Runtime dependencies in `requirements.txt`.
- Test dependencies in `requirements-dev.txt`.
- Streamlit upload limit and telemetry setting in `.streamlit/config.toml`.
- No paid APIs, external AI services, API keys, or runtime environment variables
  are required.
- `.env.example` is intentionally absent because no environment variables are
  required.

## 9. Security and data quality controls

- Raw input validation covers required columns, numeric types, binary values,
  ranges, blank categories, and empty uploads.
- Unknown categorical values are surfaced as warnings; the saved one-hot encoder
  uses `handle_unknown="ignore"`.
- CSV uploads are type-restricted and limited to 50 MB.
- Uploaded data is held in the Streamlit session; the app does not persist it.
- No credentials or paid-service clients are present in application source.
- The original raw dataset checksum is enforced by a test.
- The app does not write to the raw dataset.
- The test suite verifies no destructive raw-data calls in application source.

## 10. Limitations

- Probability estimates are not guarantees.
- Historical associations are not causal effects.
- No cost, revenue, or capacity data are available for profit optimisation.
- K-Means silhouette is modest.
- Phase 2 preprocessing was fitted before Phase 4 CV.
- The threshold is model-specific and OOF-selected.
- Concurrency, very large uploads, and visual/accessibility audits were not
  stress-tested.
