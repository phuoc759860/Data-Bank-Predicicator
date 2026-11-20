# Phase 4 - K-Means Clustering Evaluation

**Source:** Phase 3 artifacts (`models/kmeans.joblib`, `reports/phase3/clustering_metrics.csv`, `cluster_profiles.csv`).  
**Status:** Review only. K-Means was not refitted in Phase 4.  
**Selected clusters:** 4  
**Independence:** `y` and `duration` were not clustering inputs.

---

## 1. Clustering inputs

The saved K-Means pipeline expects these engineered customer features:

- `age`, `balance`, `previous`, `default`, `housing`, `loan`, `has_previous_contact`, `previous_campaign_success`, `job`, `marital`, `education`, `poutcome`

No target label and no `duration` field are present. The target is used below only as a post-hoc description of observed subscription rates.

## 2. Model selection metrics

| k | inertia | silhouette_score | min_cluster_count | min_cluster_share_pct | cluster_sizes |
| --- | --- | --- | --- | --- | --- |
| 2 | 186080.86 | 0.16 | 14,142 | 39.10 | 14142, 22026 |
| 3 | 167778.89 | 0.17 | 5,835 | 16.13 | 5835, 18268, 12065 |
| 4 | 150512.02 | 0.17 | 890 | 2.46 | 11833, 890, 5696, 17749 |
| 5 | 141582.14 | 0.14 | 801 | 2.21 | 6763, 10407, 5457, 12740, 801 |
| 6 | 135192.94 | 0.13 | 788 | 2.18 | 4891, 6843, 788, 9374, 8871, 5401 |
| 7 | 123777.29 | 0.17 | 1 | 0.00 | 11188, 5208, 1, 1944, 170, 472, 17185 |
| 8 | 115274.77 | 0.13 | 1 | 0.00 | 9248, 7477, 155, 1, 12029, 1695, 472, 5091 |
| 9 | 109175.72 | 0.13 | 1 | 0.00 | 5674, 5779, 152, 1, 9231, 1574, 471, 5067, 8219 |
| 10 | 105557.81 | 0.13 | 1 | 0.00 | 1354, 6023, 6899, 5532, 470, 5000, 87, 1, 5049, 5753 |

The selected solution is **k=4** with inertia **150512.02** and silhouette **0.1674**. k=7 has a slightly higher raw silhouette, but it creates a one-customer cluster and a 170-customer cluster; the 2% minimum cluster-share guardrail rejects that degenerate solution. k=4 provides four usable marketing segments.

Silhouette is modest because the feature space mixes scaled numeric variables with one-hot customer categories. The value of the solution is interpretability and business differentiation, not a high absolute separation score. Phase 3 also reported a held-out test silhouette of **0.1697**, close to the training value, so the segmentation generalises reasonably even though the absolute separation is modest.

## 3. Cluster sizes and characteristics

| cluster | n_customers | share_pct | age_mean | balance_median | top_job | top_job_pct | pct_housing_loan | pct_previous_contact | observed_subscription_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 11,833 | 32.72 | 52.37 | 529.00 | blue-collar | 21.63 | 41.36 | 5.97 | 10.39 |
| 1 | 890 | 2.46 | 43.03 | 12054.50 | management | 33.37 | 45.39 | 18.65 | 14.49 |
| 2 | 5,696 | 15.75 | 38.32 | 538.00 | management | 22.79 | 67.03 | 100.00 | 21.01 |
| 3 | 17,749 | 49.07 | 33.96 | 341.00 | blue-collar | 22.34 | 62.35 | 0.08 | 9.44 |

### Cluster 0 - 11,833 customers (32.7%)

This segment has mean age **52.4**, median balance **529 EUR**, most common job **blue-collar** (21.6%), housing-loan share **41.4%**, and previous-contact share **6.0%**. The post-hoc observed subscription rate is **10.39%** (-1.31 percentage points versus the overall 11.70%).

**Interpretation:** older, mid-balance customers; the most common job is blue-collar, 41% have a housing loan, and few had previous campaign contact. Conversion is close to the overall rate.

This segment is close to the overall conversion rate. It can support volume campaigns but is not the most differentiated group.

### Cluster 1 - 890 customers (2.5%)

This segment has mean age **43.0**, median balance **12,054 EUR**, most common job **management** (33.4%), housing-loan share **45.4%**, and previous-contact share **18.7%**. The post-hoc observed subscription rate is **14.49%** (+2.80 percentage points versus the overall 11.70%).

**Interpretation:** mid-age, high-balance customers with a management orientation; this small segment has above-average observed conversion and may support a premium deposit proposition.

This is an above-average segment. It is a candidate for priority outreach, a targeted term-deposit proposition, or a higher-touch channel where the expected value supports the cost.

### Cluster 2 - 5,696 customers (15.7%)

This segment has mean age **38.3**, median balance **538 EUR**, most common job **management** (22.8%), housing-loan share **67.0%**, and previous-contact share **100.0%**. The post-hoc observed subscription rate is **21.01%** (+9.32 percentage points versus the overall 11.70%).

**Interpretation:** younger, mid-balance customers; every customer in this segment had previous campaign contact and observed conversion is the highest of the four groups. This is the clearest warm-lead segment.

This is an above-average segment. It is a candidate for priority outreach, a targeted term-deposit proposition, or a higher-touch channel where the expected value supports the cost.

### Cluster 3 - 17,749 customers (49.1%)

This segment has mean age **34.0**, median balance **341 EUR**, most common job **blue-collar** (22.3%), housing-loan share **62.3%**, and previous-contact share **0.1%**. The post-hoc observed subscription rate is **9.44%** (-2.26 percentage points versus the overall 11.70%).

**Interpretation:** youngest, lowest-balance customers; almost none had previous campaign contact and observed conversion is below average. Expensive high-touch outreach is harder to justify here.

This is a below-average segment. Prefer lower-cost channels, self-service campaigns, or a different product proposition rather than expensive high-touch calls.

## 4. Business interpretation

- **Cluster 2** is the clearest priority segment: it combines prior contact history with a substantially above-average observed subscription rate. It should be considered for warm-lead outreach.
- **Cluster 1** is small but high-balance and management-oriented; it may justify a premium deposit offer, subject to campaign economics.
- **Cluster 3** is the largest and youngest/lowest-balance group. It converts below average, so expensive calls are harder to justify.
- **Cluster 0** is older and near-average in conversion; it can support volume campaigns with disciplined call scripting.

These segments should be combined with the supervised probability ranking: first rank customers by predicted conversion probability, then use segment membership to choose channel, offer, and contact intensity. Clustering does not replace the conversion model.

## 5. Limitations

- K-Means assumes roughly spherical, similarly sized clusters and uses Euclidean distance on one-hot dimensions.
- The silhouette is modest; cluster boundaries are not sharp and membership should be treated as a marketing heuristic.
- The observed subscription rate is descriptive and not causal. It was not used to create the clusters.
- Cluster IDs are arbitrary; business names should be agreed with marketing stakeholders.

## 6. Reference charts

The Phase 3 clustering charts remain the source of truth:

- `reports/phase3/customer_clusters.png`
- `reports/phase3/elbow_method.png`
- `reports/phase3/silhouette_scores.png`
