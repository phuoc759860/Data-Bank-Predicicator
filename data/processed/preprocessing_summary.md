# Phase 2 - Data Preprocessing and Preparation

**Project:** Bank Marketing Intelligence System  
**Course context:** Information Systems Management  
**Generated:** 2026-11-17 17:45:10  
**Input:** `data/raw/bank-full.csv`  
**Raw file SHA-256 (verified unchanged):** `d1513ec63b385506...`  
**Target:** `y` (1 = subscribed a term deposit, 0 = did not)  
**Split:** 80% train / 20% test, stratified, random_state=42  
**Final matrices:** X_train 36,168 x 59, X_test 9,043 x 59  
**Scope:** preprocessing only - no machine-learning model was trained.

---

## 1. Executive summary

- The original **45,211 x 17** dataset was read with `pd.read_csv(path, sep=';')` and preserved byte-for-byte; the SHA-256 checksum was verified after processing.
- No NULL cells and **0 duplicate rows** were found. No rows were deleted anywhere in Phase 2.
- 'unknown' categories were **retained as explicit levels** rather than imputed or dropped. They affect **4 categorical columns**, most notably `poutcome` and `contact`.
- `pdays = -1` was converted into the binary feature `has_previous_contact` and a sentinel-free recency feature; **36,954 records (81.74%)** had no previous contact.
- **7 numerical variables** were profiled for outliers. Extreme values were retained; only scaling is applied, fitted on training data.
- Data were split **before** fitting any transformer. The fitted pipeline uses OneHotEncoder for nominal variables, 0/1 passthrough for binary variables, and StandardScaler for numeric variables.
- **19 source features** (including 4 engineered features) expand to **59 model-ready columns**: 6 scaled numeric, 5 binary, and 48 one-hot indicators.
- `duration` is excluded from the primary pre-call model because it is recorded only after the call and would leak the outcome.
- Post-transformation validation: **0 missing values, 0 non-finite values**, train positive rate 11.70%, test positive rate 11.70% (overall 11.70%).

---

## 2. Data cleaning (Task 1)

### 2.1 Input integrity

The source file is opened read-only. It is never overwritten, and the script re-computes its SHA-256 checksum after writing all outputs to prove that the original data remain unchanged.

- Rows: **45,211**; columns: **17**
- Missing cells: **0**
- Exact duplicate rows: **0**
- `pdays = -1`: **36,954 (81.74%)**

### 2.2 'unknown' category investigation and decisions

The table compares the subscription rate of 'unknown' records with records that have a known value. This tests whether 'unknown' is random or carries behavioural information.

| column | unknown_count | unknown_pct | unknown_rate_pct | known_rate_pct | gap_pp |
| --- | --- | --- | --- | --- | --- |
| job | 288 | 0.64 | 11.81 | 11.70 | 0.11 |
| marital | 0 | 0.00 |  | 11.70 |  |
| education | 1,857 | 4.11 | 13.57 | 11.62 | 1.95 |
| default | 0 | 0.00 |  | 11.70 |  |
| housing | 0 | 0.00 |  | 11.70 |  |
| loan | 0 | 0.00 |  | 11.70 |  |
| contact | 13,020 | 28.80 | 4.07 | 14.78 | -10.71 |
| month | 0 | 0.00 |  | 11.70 |  |
| poutcome | 36,959 | 81.75 | 9.16 | 23.06 | -13.90 |

**Decision:** no row is deleted and no 'unknown' value is replaced by the mode. The categories are kept as explicit one-hot levels so the model can learn their effect, and future unseen categories are handled with `handle_unknown='ignore'`. For `poutcome`, the additional `previous_campaign_success` feature isolates confirmed prior success. This preserves the full sample and does not fabricate a customer attribute.

### 2.3 pdays = -1 handling

`pdays = -1` appears in **36,954 records (81.74%)** and means *never contacted before*, not *one day before*. Using it as a number would distort averages and distances. The following were created:

- `has_previous_contact`: 1 when `pdays != -1`, otherwise 0.
- `pdays_since_previous_contact`: 0 for never-contacted clients; otherwise the original positive day count. The binary flag distinguishes the sentinel from genuine recency.

The consistency check confirms that `pdays = -1` and `previous = 0` agree for all records (mismatches: **0**).

### 2.4 Numerical outlier investigation

Outliers were profiled with the 1.5 x IQR rule and tail percentiles. They are **not deleted, capped, or winsorised**, because high balances and long campaign histories are legitimate banking observations and may be predictive.

| variable | outlier_count | outlier_pct | min | p1 | median | p99 | max | treatment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 487 | 1.08 | 18.00 | 23.00 | 39.00 | 71.00 | 95.00 | Retained (no deletion, capping, or winsorising). |
| balance | 4,729 | 10.46 | -8019.00 | -627.00 | 448.00 | 13164.90 | 102127.00 | Retained (no deletion, capping, or winsorising). |
| day | 0 | 0.00 | 1.00 | 2.00 | 16.00 | 31.00 | 31.00 | Retained (no deletion, capping, or winsorising). |
| duration | 3,235 | 7.16 | 0.00 | 11.00 | 180.00 | 1269.00 | 4918.00 | Retained (no deletion, capping, or winsorising). |
| campaign | 3,064 | 6.78 | 1.00 | 1.00 | 2.00 | 16.00 | 63.00 | Retained (no deletion, capping, or winsorising). |
| pdays | 8,257 | 18.26 | -1.00 | -1.00 | -1.00 | 370.00 | 871.00 | Retained (no deletion, capping, or winsorising). |
| previous | 8,257 | 18.26 | 0.00 | 0.00 | 0.00 | 8.90 | 275.00 | Retained (no deletion, capping, or winsorising). |

Balance ranges from -8,019 to 102,127 EUR and includes **3,766 negative balances** (8.33%). `StandardScaler` is sensitive to extreme values, so Phase 3 should compare scaled linear models with tree-based models and consider robust alternatives if performance is unstable. Any alternative transformation must be refitted inside the training folds.

The high outlier counts for `pdays` and `previous` reflect the campaign-history structure rather than data errors: most clients have no previous contact, so any positive history appears far above the median. The `-1` sentinel is handled separately below; `duration` is excluded from modelling regardless.

---

## 3. Feature engineering (Task 2)

All engineering rules are fixed constants applied independently to each split; no aggregate statistic from the training or test set is used.

| feature | source | purpose |
| --- | --- | --- |
| age_group | age | Captures the non-linear age effect found in Phase 1 (65+ and 18-24 convert far above the 35-54 core). |
| has_previous_contact | pdays | Separates clients with campaign history from first-time contacts and removes the -1 sentinel ambiguity. |
| contact_frequency_group | campaign | Groups current-campaign contact intensity (1, 2-3, 4-5, 6+) to represent persistence and possible call fatigue. |
| previous_campaign_success | poutcome | Flags warm leads whose previous campaign succeeded, the strongest descriptive signal in Phase 1 (64.7% conversion). |
| pdays_since_previous_contact | pdays | Recency of the previous contact without the -1 sentinel; 0 is interpreted together with has_previous_contact. |
| default / housing / loan | categorical yes/no | Mapped to 0/1 so binary variables use one column rather than two redundant one-hot columns. |

---

## 4. Encoding (Task 3)

| Group | Source features | Method | Output columns |
| --- | --- | --- | --- |
| Nominal | `job`, `marital`, `education`, `contact`, `month`, `poutcome`, `age_group`, `contact_frequency_group` | OneHotEncoder(handle_unknown='ignore') | 48 |
| Binary | `default`, `housing`, `loan`, `has_previous_contact`, `previous_campaign_success` | 0/1 passthrough (yes/no mapped) | 5 |
| Numeric | `age`, `balance`, `day`, `campaign`, `previous`, `pdays_since_previous_contact` | StandardScaler | 6 |

One-hot encoding is applied only to nominal variables. Every level is kept (no drop='first') to preserve interpretability and avoid assuming an order. The final column names retain transformer prefixes (`num__`, `bin__`, `cat__`) and the original variable/category, e.g. `cat__job_blue-collar`. All 59 names are documented in `feature_dictionary.csv`.

---

## 5. Scaling and leakage prevention (Task 4)

- The raw data were split **first**; feature engineering uses fixed rules only.
- The pipeline was fitted on **36,168 training rows** and then applied to the **9,043 test rows**.
- `StandardScaler` therefore learned its means and standard deviations from training data only. Test data never influenced scaling, category discovery, or any other fitted parameter.
- Binary variables are not scaled, and one-hot indicators remain 0/1.
- The fitted pipeline is saved as `preprocessor.joblib` so Phase 3 can apply exactly the same transformation to new data.
- For cross-validation in Phase 3, the preprocessing pipeline must be fitted inside each training fold; resampling, if used, must also occur inside folds.

---

## 6. Feature selection and duration exclusion (Task 5)

The selected source-level feature set contains **19 features** before encoding:

- **Client Profile:** `age`, `balance`, `default`, `housing`, `loan`, `job`, `marital`, `education`
- **Current Campaign:** `day`, `contact`, `month`
- **Campaign History:** `campaign`, `previous`, `pdays_since_previous_contact`, `poutcome`
- **Engineered:** `has_previous_contact`, `previous_campaign_success`, `age_group`, `contact_frequency_group`

### Why `duration` is excluded

`duration` is the length of the last phone call and is recorded only after the call ends. At the moment a customer is selected for a pre-call campaign, this value does not exist. Phase 1 found a strong raw correlation with the target (+0.395), so including it would produce optimistic, unreproducible results through **target leakage**. It is therefore removed before the pipeline and marked `selected_for_pre_call_model = False` in the feature dictionary. It may be used only in a separate call-time or post-call use case, never in this pre-call targeting dataset.

`pdays` is also not passed through in its raw form; it is replaced by `has_previous_contact` and `pdays_since_previous_contact` as described above.

---

## 7. Train/test split (Task 6)

- Method: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
- **Training set:** 36,168 rows (80.00%)
- **Test set:** 9,043 rows (20.00%)
- **Features per split:** 59
- Positive class: train 4,231 (11.70%), test 1,058 (11.70%), overall 11.70%

| split | class | count | percentage | split_total |
| --- | --- | --- | --- | --- |
| train | no | 31,937 | 88.30 | 36,168 |
| train | yes | 4,231 | 11.70 | 36,168 |
| test | no | 7,985 | 88.30 | 9,043 |
| test | yes | 1,058 | 11.70 | 9,043 |
| overall | no | 39,922 | 88.30 | 45,211 |
| overall | yes | 5,289 | 11.70 | 45,211 |

Stratification preserves the 11.70% subscription rate in both splits, so evaluation in Phase 3 is not distorted by a different class mix.

---

## 8. Validation results (Task 7)

| Check | Result |
| --- | --- |
| X_train shape | 36,168 x 59 |
| X_test shape | 9,043 x 59 |
| Missing values after transformation | 0 |
| Non-finite values | 0 |
| Target values | 0 and 1 only |
| Train + test rows equal original | 45,211 = 45,211 |
| Train / test positive rate | 11.70% / 11.70% |
| Raw file checksum unchanged | Yes |

---

## 9. Saved artefacts

All files are written to `data/processed/`:

- `X_train.csv` - scaled/encoded training features
- `X_test.csv` - scaled/encoded test features
- `y_train.csv` - training target (0/1)
- `y_test.csv` - test target (0/1)
- `preprocessing_summary.md` - this document
- `feature_dictionary.csv` - all 59 model columns plus the excluded `duration`
- `target_distribution_after_split.csv` - class counts and percentages
- `unknown_values_investigation.csv` - unknown counts and conversion comparison
- `outlier_investigation.csv` - IQR/tail outlier profile
- `preprocessor.joblib` - fitted scikit-learn Pipeline

The processed CSVs contain the **transformed** matrices, not raw categories. Use the feature dictionary and the saved pipeline to interpret or reuse them.

---

## 10. Limitations and Phase 3 notes

- **No model has been trained or evaluated.** These matrices are inputs for Phase 3 only.
- **Class imbalance remains** at ~11.7% positives. Use recall, precision, F1, ROC-AUC/PR-AUC, lift, and cost-based evaluation rather than accuracy.
- **'unknown' is retained as information**, but its meaning may differ across fields (not collected vs no previous campaign). Treat category effects with care when explaining models.
- **Outliers are retained.** StandardScaler can be sensitive to them; compare with tree-based models and consider robust scaling only if justified by validation performance.
- **Sentinel recency:** `pdays_since_previous_contact = 0` is only meaningful together with `has_previous_contact`; do not interpret 0 as a same-day call.
- **No year field** is available, so long-term time effects cannot be modelled.
- **Feature redundancy:** `previous_campaign_success` is derived from `poutcome`; this is intentional for interpretability but can be revisited for linear models with regularisation.
- **Scaling must stay inside the CV pipeline** in Phase 3 to avoid leakage.

---

## 11. Reproduction

```bash
python src/phase2_preprocessing.py
```

Optional arguments:

```bash
python src/phase2_preprocessing.py \
    --data data/raw/bank-full.csv \
    --output-dir data/processed \
    --test-size 0.2 \
    --random-state 42
```

The script validates the schema and target, fits transformations on training data only, verifies zero missing/non-finite values, and confirms that the raw CSV checksum is unchanged.
