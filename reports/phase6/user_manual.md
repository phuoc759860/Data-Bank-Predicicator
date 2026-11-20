# Phase 6 — User Manual

## 1. What the application does

The Bank Marketing Intelligence app helps a marketing team:

- Explore the actual customer base and historical campaign outcomes.
- Score an individual customer's likelihood of subscribing to a term deposit.
- Score an uploaded customer list and download results.
- Review descriptive customer segments from K-Means.
- Compare observed subscription rates across customer and campaign dimensions.

The application is a decision-support tool. It does not guarantee that a customer
will subscribe, and descriptive rates do not prove that changing a customer
attribute or channel would cause a subscription.

## 2. Starting the application

From the project root:

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pip install -r requirements.txt
python -m streamlit run app/Home.py
```

Open the URL printed by Streamlit, normally:

```text
http://localhost:8501
```

Use the sidebar to move between pages. The pages are:

1. Home
2. Executive Dashboard
3. Customer Prediction
4. Batch Prediction
5. Customer Segmentation
6. Marketing Analytics

## 3. Home page

The Home page displays:

- Total raw customers.
- Observed subscribers.
- Overall observed subscription rate.
- Selected model test F1.
- Status of the dataset, prediction model, and segmentation artifacts.
- A summary of the selected Random Forest and Phase 4 threshold.

If an essential artifact is missing or malformed, the app stops and shows an
error instead of silently substituting another model.

## 4. Executive Dashboard

### 4.1 Filters

The sidebar provides filters for:

- Job
- Education
- Marital status
- Contact channel
- Age range

An empty multiselect means "all values". The page reports how many customers are
currently in view.

### 4.2 Charts

- **Subscription distribution:** observed Yes/No counts.
- **Age distribution:** customer counts by age, split by observed outcome.
- **Job and education:** customer counts by category.
- **Campaign performance:**
  - Subscription rate by contact channel.
  - Subscription rate by previous campaign outcome.
  - Subscription rate by contact month.
  - Subscription rate by current-campaign contact frequency.

Group labels include sample size so a very small group is not read as a stable
business signal.

### 4.3 Interpretation

Use the dashboard to form questions, not to claim causality. A high observed rate
for a job or month may reflect other customer and campaign factors.

## 5. Customer Prediction

### 5.1 Input fields

The form collects pre-call customer and campaign-history fields:

| Field | Notes |
|---|---|
| Age | 18–120 in the form; validation enforces this range |
| Job | Observed dataset categories |
| Marital status | Observed categories |
| Education | Observed categories |
| Credit in default? | no / yes |
| Average yearly balance | EUR |
| Housing loan? | no / yes |
| Personal loan? | no / yes |
| Contact channel | cellular / telephone / unknown |
| Last contact month | Observed months |
| Last contact day of month | 1–31 |
| Contacts in current campaign | At least 1 |
| Contacts before this campaign | At least 0 |
| Days since previous contact | `-1` means never contacted |
| Previous campaign outcome | failure / other / success / unknown |

`duration` is deliberately absent. It is only known after a call and would leak
post-contact information into the prediction.

### 5.2 Result

After submitting, the page displays:

- Estimated probability of subscription.
- Selected Phase 4 threshold (`0.20`).
- **Likely to subscribe** when probability is at or above the threshold.
- **Below the decision threshold** otherwise.
- A probability bar and an interpretation warning.

The probability is a model estimate from historical patterns. It is not a
guarantee. Use it to prioritise outreach with human review and campaign capacity.

## 6. Batch Prediction

### 6.1 Required CSV columns

The uploaded CSV must contain these raw pre-call columns:

```text
age, job, marital, education, default, balance, housing, loan, contact,
day, month, campaign, pdays, previous, poutcome
```

Column names are case-insensitive. The file may also contain:

- `duration` — optional; ignored.
- `y` — optional; ignored.

### 6.2 Accepted values

- Numeric: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.
- Binary: `default`, `housing`, `loan` as `yes`/`no` or `1`/`0`.
- Categorical: observed dataset values for `job`, `marital`, `education`,
  `contact`, `month`, `poutcome`.
- `pdays = -1` means no previous contact.
- Age must be 18–120.
- Day must be 1–31.
- Campaign must be at least 1.
- Previous must be at least 0.
- `pdays` must be `-1` or non-negative.

Unknown categorical values produce a warning. The saved one-hot encoder handles
them with `handle_unknown="ignore"`, but those rows may have weaker evidence.

### 6.3 Output

The preview and downloaded CSV preserve the input columns and append:

| Column | Meaning |
|---|---|
| `predicted_probability` | Model score in `[0, 1]` |
| `predicted_subscription` | `Yes` or `No` at threshold `0.20` |
| `predicted_label` | `1` or `0` |
| `decision_threshold` | `0.20` |
| `model_name` | Selected model name |

Use the probability to rank customers. Use the predicted label as an operational
default, not a guarantee.

### 6.4 Download

Click **Download predictions as CSV**. The app exports the full scored batch as
`bank_marketing_predictions.csv`.

### 6.5 Common errors

- **The uploaded file is empty:** upload a non-empty CSV.
- **Missing required columns:** add the listed columns.
- **Column must contain numeric values:** correct non-numeric fields.
- **Column must be yes/no (or 1/0):** correct binary fields.
- **Invalid values in ...:** correct the listed range violation.
- **Blank category values:** fill missing categorical fields.

The app does not generate predictions when validation fails.

## 7. Customer Segmentation

This page uses the real Phase 3 K-Means pipeline and the Phase 2 training split.

### 7.1 What is shown

- Training customer count (`36,168`).
- Number of segments (`4`).
- Training silhouette score and inertia.
- Cluster distribution.
- Observed subscription rate by segment.
- A PCA scatter plot for display only.
- Actual segment profiles:
  - Customers and share.
  - Mean age and median balance.
  - Most common job.
  - Housing-loan share.
  - Previous-contact share.
  - Observed subscription rate.

### 7.2 How to use it

Use segments to choose channel, offer, and contact intensity after ranking
customers with the supervised prediction model. Segments describe customer
profiles; they do not use the subscription label to form clusters.

The K-Means silhouette is modest, so treat segments as exploratory rather than
definitive customer identities.

## 8. Marketing Analytics

The page compares observed subscription rates by:

- Job
- Age group
- Education
- Marital status
- Previous campaign outcome
- Contact channel
- Contact month
- Current-campaign contact frequency

Use the sidebar filters and the minimum-group-size slider. The minimum group size
prevents tiny groups from dominating the charts.

The **Evidence and caveats** tab includes:

- Dynamic observed patterns.
- A group-summary table.
- An explicit warning that correlation is not causation.

Use these views to design controlled tests, not to assert that changing a factor
will cause a subscription.

## 9. Data and privacy notes

- The app uses `data/raw/bank-full.csv` and does not modify it.
- Uploaded CSV data is held in the Streamlit session and is not persisted by the
  app.
- Uploads are restricted to CSV and limited to 50 MB by `.streamlit/config.toml`.
- The app contains no API keys, passwords, paid API calls, or external AI service
  calls.
- No environment variables are required.

## 10. Troubleshooting

| Problem | Action |
|---|---|
| `streamlit` command not found | Use `python -m streamlit run app/Home.py` |
| Missing model/dataset error | Verify `models/best_model.joblib`, `models/kmeans.joblib`, and `data/raw/bank-full.csv` exist |
| Port already in use | Use `--server.port 8502` or another free port |
| CSV rejected | Compare the header and values with the required format above |
| Slow first load | The Random Forest artifact is large; initial model loading can take a moment |
| Page shows no data | Widen filters or reset the minimum group size |

## 11. Scope

This manual covers the local Phase 6 application. It does not describe Phase 7 or
any production cloud deployment.
