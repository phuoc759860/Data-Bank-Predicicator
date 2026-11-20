# Phase 3 - Data Mining Model Development

**Project:** Bank Marketing Intelligence System  
**Course context:** Information Systems Management  
**Generated:** 2026-11-17 19:38:00  
**Models:** Decision Tree, Random Forest, K-Means  
**Scope:** Phase 3 only - no Phase 4/5 work and no test-set tuning.

---

## 1. Executive summary

- **Decision Tree** (depth=8, leaves=121): F1=0.4417, recall=0.5803, precision=0.3566, ROC-AUC=0.7787.
- **Random Forest** (best params in report): F1=0.4862, recall=0.5246, precision=0.4531, ROC-AUC=0.8031.
- Both classifiers reuse the exact Phase 2 stratified split (36,168 train / 9,043 test) and exclude `duration`.
- Hyperparameters were selected with 5-fold stratified CV on training data only; the test set was evaluated once per model.
- **K-Means** selected **k=4** from k=2-10 using silhouette score with a minimum cluster-share guardrail (2% of training customers). The target `y` and `duration` were not clustering inputs.
- The processed features are reused from Phase 2; the saved preprocessing pipeline is copied to `models/preprocessing_pipeline.joblib`.

---

## 2. Data and leakage protection

- Inputs: `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` from Phase 2.
- The Phase 2 fitted pipeline was trained on the training split only. It was loaded and reused for classification, not refitted on test data.
- Model selection used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training set.
- `duration` is absent from the 59 model-ready Phase 2 features, so the pre-call models cannot exploit post-call leakage.
- K-Means has its own `StandardScaler` + `OneHotEncoder` pipeline fitted on training customer rows only. `y` is used solely after clustering to describe observed conversion, never to form clusters.
- The raw Phase 2 split was reproduced and verified against the saved `y_train.csv` before clustering, ensuring the same customers are used.

## 3. Model selection

Hyperparameters were selected by 5-fold stratified cross-validation on the training set, using F1 for the positive class as the selection metric. The test set is not involved in this table.

| model | best_params | cv_f1_mean | cv_f1_std | test_f1 | test_roc_auc |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | {"max_depth": 8, "min_samples_leaf": 25, "min_samples_split": 2} | 0.42 | 0.02 | 0.44 | 0.78 |
| Random Forest | {"max_depth": null, "min_samples_leaf": 5, "n_estimators": 200} | 0.47 | 0.01 | 0.49 | 0.80 |

---

## 4. Test-set evaluation and model comparison

All values below are from the held-out test set. The test set was not used for hyperparameter selection or threshold tuning.

| model | accuracy | precision | recall | f1 | roc_auc | true_positive | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | 0.83 | 0.36 | 0.58 | 0.44 | 0.78 | 614 | 1,108 | 444 |
| Random Forest | 0.87 | 0.45 | 0.52 | 0.49 | 0.80 | 555 | 670 | 503 |

### Which model performs better overall?

**Random Forest** has the higher F1-score (0.4862) and **Random Forest** has the higher ROC-AUC (0.8031). Random Forest also has the higher raw accuracy (0.8703), but accuracy is not the primary criterion because 88.30% of customers do not subscribe. Random Forest is the stronger ranked classifier on this test set.

### Which model identifies more actual subscribers?

**Decision Tree** has the higher recall (0.5803), capturing 614 of the 1,058 actual subscribers in the test set. The Decision Tree captured 614; the Random Forest captured 555. The model with higher recall is preferable when the bank wants to avoid missing interested customers and can absorb more calls.

The Decision Tree finds 59 more subscribers than the Random Forest, but it also predicts 497 more positives, i.e. extra calls. The Random Forest finds 90.4% of the Decision Tree's subscribers while making 28.9% fewer predicted contacts. That is usually the better cost trade-off unless call capacity is effectively unconstrained.

### Which model has better precision?

**Random Forest** has the higher precision (0.4531). Precision here means the share of predicted subscribers who actually subscribed: Decision Tree 0.3566 vs Random Forest 0.4531. Higher precision reduces wasted call-centre effort; higher recall reduces missed revenue opportunities. The choice depends on campaign economics.

Relative to the 11.70% positive base rate, precision lift is **3.05x** for the Decision Tree and **3.87x** for the Random Forest. A value above 1 means the model enriches the contact list compared with calling customers at random.

### Which is more suitable for the bank's marketing objectives?

For a broad, budget-constrained campaign, **Random Forest** is the more balanced choice because it has the best F1 and ROC-AUC, which indicates better ranking of likely subscribers across thresholds. **Decision Tree** is preferable if the bank prioritises contacting as many potential subscribers as possible and has capacity for extra calls. If call costs dominate, the higher-precision model should be preferred, possibly with a threshold selected to match the contact budget. Accuracy should not decide the campaign: a constant 'no' model would score 88.30% and identify zero subscribers, which is worse for marketing than either evaluated model.

### Trade-offs: complexity vs interpretability

- **Decision Tree:** selected with {"max_depth": 8, "min_samples_leaf": 25, "min_samples_split": 2}. It can be read as explicit if/then rules, is easy to explain to marketing and compliance stakeholders, and exposes clear customer conditions. It is less stable and generally weaker in predictive performance because a single tree overfits or underfits more easily.
- **Random Forest:** selected with {"max_depth": null, "min_samples_leaf": 5, "n_estimators": 200}. It averages many trees, giving stronger discrimination and more stable probabilities, but its internal logic is much harder to explain. It provides feature importance for global interpretation, while local explanations such as SHAP would be needed for individual decisions.
- The Random Forest is the better candidate for a production ranking model; the Decision Tree is the better candidate for a transparent policy or a first explanation of which customer characteristics matter. This is a business trade-off, not only a metric trade-off.

### Important imbalance note

Both models use `class_weight='balanced'`. At the default 0.5 threshold, the Decision Tree predicts 1,722 subscribers (19.04% of test rows) and the Random Forest predicts 1,225 (13.55%), compared with the 11.70% base rate. This deliberately trades precision for recall. The relevant business question is how many real subscribers are found at an acceptable cost per contact; accuracy alone hides this trade-off.

---

## 5. Decision Tree interpretation

The selected tree has **8 levels** and **121 leaves**. The chart shows the top 3 levels at readable scale; the complete tree is in `decision_tree_report.txt`, and all 121 leaf rules are in `decision_rules.csv`.

### Key decision rules (ranked by subscribers captured in training)

| rank | rule | samples | subscribers | precision_pct | recall_of_all_subscribers_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | poutcome != success AND contact != unknown AND housing = 0 AND month != aug AND month != jul AND month != nov AND month != jan AND month != feb | 2,630 | 939 | 35.70 | 22.19 |
| 2 | poutcome != success AND contact != unknown AND housing = 1 AND pdays_since_previous_contact <= 374.50 AND month != jun AND month != mar AND month != oct AND marital != single | 8,738 | 590 | 6.75 | 13.94 |
| 3 | poutcome != success AND contact != unknown AND housing = 1 AND pdays_since_previous_contact <= 374.50 AND month != jun AND month != mar AND month != oct AND marital = single | 3,238 | 320 | 9.88 | 7.56 |
| 4 | poutcome != success AND contact != unknown AND housing = 0 AND month != aug AND month != jul AND month != nov AND month != jan AND month = feb | 1,101 | 195 | 17.71 | 4.61 |
| 5 | poutcome != success AND contact != unknown AND housing = 0 AND month = aug AND contact_frequency_group != 1 AND day <= 18.50 AND pdays_since_previous_contact <= 23.00 AND age <= 59.50 | 1,490 | 132 | 8.86 | 3.12 |
| 6 | poutcome != success AND contact != unknown AND housing = 0 AND month != aug AND month = jul AND pdays_since_previous_contact <= 82.50 AND day > 6.50 AND marital != married | 1,043 | 116 | 11.12 | 2.74 |
| 7 | poutcome != success AND contact != unknown AND housing = 0 AND month != aug AND month = jul AND pdays_since_previous_contact <= 82.50 AND day > 6.50 AND marital = married | 1,431 | 78 | 5.45 | 1.84 |
| 8 | poutcome = success AND housing = 0 AND pdays_since_previous_contact > 80.50 AND pdays_since_previous_contact <= 98.50 AND job != technician AND day > 7.50 AND day <= 13.50 | 84 | 74 | 88.10 | 1.75 |
| 9 | poutcome != success AND contact != unknown AND housing = 0 AND month != aug AND month != jul AND month = nov AND day > 16.50 AND age <= 58.50 | 1,080 | 72 | 6.67 | 1.70 |
| 10 | poutcome != success AND contact = unknown AND marital != married AND day <= 29.50 AND month != may AND age > 30.50 AND campaign <= 10.50 AND loan = 0 | 938 | 59 | 6.29 | 1.39 |

### Highest-precision rules (minimum 50 training samples)

| rule | samples | subscribers | precision_pct |
| --- | --- | --- | --- |
| poutcome = success AND housing = 0 AND pdays_since_previous_contact > 80.50 AND pdays_since_previous_contact <= 98.50 AND job != technician AND day > 7.50 AND day <= 13.50 | 84 | 74 | 88.10 |
| poutcome = success AND housing = 0 AND pdays_since_previous_contact > 80.50 AND pdays_since_previous_contact <= 98.50 AND job != technician AND day > 18.50 | 73 | 57 | 78.08 |
| poutcome = success AND housing = 0 AND pdays_since_previous_contact > 80.50 AND pdays_since_previous_contact <= 98.50 AND job != technician AND day <= 7.50 | 68 | 51 | 75.00 |
| poutcome != success AND contact != unknown AND housing = 1 AND pdays_since_previous_contact <= 374.50 AND month != jun AND month != mar AND month = oct AND day > 16.50 | 53 | 28 | 52.83 |
| poutcome != success AND contact != unknown AND housing = 0 AND month != aug AND month != jul AND month = nov AND day <= 16.50 AND pdays_since_previous_contact <= 95.50 | 121 | 54 | 44.63 |

Example: the top-precision leaf captures **74 subscribers from 84 customers (88.1% precision)**. High-precision rules are better for small, targeted campaigns; high-volume rules are better for broad outreach.

Rules use original units where possible: numeric thresholds were converted back from the standardized scale using the Phase 2 scaler, and one-hot splits are expressed as category membership. Repeated numeric bounds are simplified for readability; exact paths remain in the full text tree.

### Top features

Feature importances for both models are in `feature_importance.csv` and `feature_importance.png`.

Random Forest top features:

| feature | importance |
| --- | --- |
| num__balance | 0.11 |
| num__age | 0.10 |
| num__day | 0.09 |
| bin__previous_campaign_success | 0.04 |
| cat__contact_unknown | 0.04 |
| bin__housing | 0.04 |
| num__pdays_since_previous_contact | 0.04 |
| cat__poutcome_success | 0.04 |
| num__campaign | 0.04 |
| cat__contact_cellular | 0.03 |

---

## 6. K-Means customer segmentation

- Candidate k values: 2-10.
- Selected **k=4** by highest silhouette among solutions where every cluster contains at least 2% of training customers. This guardrail prevents technically good but degenerate one-customer clusters.
- Full metrics and segment profiles are in `clustering_report.md`, `clustering_metrics.csv`, and `cluster_profiles.csv`.

| k | inertia | silhouette_score | min_cluster_share_pct |
| --- | --- | --- | --- |
| 2 | 186080.86 | 0.16 | 39.10 |
| 3 | 167778.89 | 0.17 | 16.13 |
| 4 | 150512.02 | 0.17 | 2.46 |
| 5 | 141582.14 | 0.14 | 2.21 |
| 6 | 135192.94 | 0.13 | 2.18 |
| 7 | 123777.29 | 0.17 | 0.00 |
| 8 | 115274.77 | 0.13 | 0.00 |
| 9 | 109175.72 | 0.13 | 0.00 |
| 10 | 105557.81 | 0.13 | 0.00 |

Cluster sizes on the training set:

| cluster | n_customers | share_pct | observed_subscription_rate_pct |
| --- | --- | --- | --- |
| 0 | 11,833 | 32.72 | 10.39 |
| 1 | 890 | 2.46 | 14.49 |
| 2 | 5,696 | 15.75 | 21.01 |
| 3 | 17,749 | 49.07 | 9.44 |

Held-out test assignment distribution (descriptive check only):

| cluster | n_customers | share_pct |
| --- | --- | --- |
| 0 | 3,026 | 33.46 |
| 1 | 234 | 2.59 |
| 2 | 1,447 | 16.00 |
| 3 | 4,336 | 47.95 |

The observed subscription rate is **post-hoc** information and was not an input to K-Means. It shows whether segments differ in business value.

---

## 7. Verification, issues, and limitations

- Both classifiers loaded from joblib reproduce identical test predictions; the preprocessing pipeline is also reloadable.
- Zero missing/non-finite values were present in the Phase 2 matrices.
- K-Means input feature list is `age`, `balance`, `previous`, `default`, `housing`, `loan`, `has_previous_contact`, `previous_campaign_success`, `job`, `marital`, `education`, `poutcome`; neither `y` nor `duration` is included.
- The test set remains untouched for final evaluation and is preserved for Phase 5.

### Issues found during implementation

- **Class-weighted tree values:** scikit-learn stores normalized, class-weighted proportions in `tree_.value` when `class_weight='balanced'` is used. Rule counts are therefore computed from actual training rows routed to each leaf via `model.apply()`, not from `tree_.value`.
- **Tree size:** the selected tree has 121 leaves, so plotting it in full is unreadable. The chart shows the top 3 levels, while the full structure and all leaf rules are exported as text/CSV.
- **Degenerate clustering solutions:** k=7 had the highest raw silhouette (0.171), but it produced a one-customer cluster and a 170-customer cluster. The minimum 2% cluster-share guardrail excludes it; k=4 has the best valid silhouette (0.167) and interpretable segment sizes.
- **CV ranking:** an initial results-export line sorted on a non-existent `rank` column; it was corrected to `rank_test_score`. This affected only the exported CV CSV, not model selection.

### Limitations

- Class imbalance remains: test positives are 1,058 of 9,043 (11.70%). Precision/recall trade-offs must be tied to campaign cost and capacity.
- Hyperparameters were selected on F1 at the default 0.5 decision threshold. A tuned threshold or precision-at-budget objective may better match the bank's economics.
- Phase 2 scaling/encoding was fitted on the whole training set before CV; this is a minor optimistic bias for CV estimates but does not touch the test set. In production retraining, keep preprocessing inside each fold or refit on the final training set.
- Tree feature importance is impurity-based and can favor continuous variables; it is not causal evidence.
- K-Means assumes roughly spherical clusters and treats one-hot dimensions as Euclidean distances. Segments are descriptive, not predictive, and silhouette scores are modest.
- No campaign cost/revenue data are available, so model selection cannot be expressed directly in profit.

---

## 8. Reproduce

```bash
python src/phase3_models.py
```

Optional:

```bash
python src/phase3_models.py \
    --processed-dir data/processed \
    --models-dir models \
    --reports-dir reports/phase3
```

The script validates inputs, trains all three models, verifies reloading, and writes the reports, charts, and model artefacts listed below.

### Artefacts

- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/kmeans.joblib`
- `models/preprocessing_pipeline.joblib`
- `reports/phase3/classification_comparison.csv`
- `reports/phase3/decision_tree_report.txt`
- `reports/phase3/random_forest_report.txt`
- `reports/phase3/clustering_report.md`
- `reports/phase3/feature_importance.csv`
- `reports/phase3/cluster_profiles.csv`
- `reports/phase3/decision_rules.csv`
- `reports/phase3/clustering_metrics.csv`
- `reports/phase3/phase3_report.md`
- `reports/phase3/*.png (8 charts)`
