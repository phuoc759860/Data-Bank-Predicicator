# Phase 6 — Installation and Local Deployment Guide

## 1. Scope

This guide prepares the project for local deployment on Windows using PowerShell.
The application runs entirely locally and uses:

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Joblib

No paid APIs, external AI services, cloud credentials, or environment variables
are required.

## 2. Tested environment

| Item | Tested value |
|---|---|
| Operating system | Windows |
| Shell | PowerShell |
| Python | 3.14.6 |
| Streamlit | 1.62.0 |
| pandas | 2.3.3 |
| NumPy | 2.3.3 |
| scikit-learn | 1.7.2 |
| Plotly | 6.9.0 |
| joblib | 1.5.2 |
| Matplotlib | 3.10.7 (Phase 1–4 reproduction) |
| Seaborn | 0.13.2 (Phase 1–4 reproduction) |
| pytest | 9.1.1 |

Use a compatible Python installation. The project was verified with the versions
above.

## 3. Required project contents

From the project root, confirm these exist:

```text
data/raw/bank-full.csv
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
models/best_model.joblib
models/best_model_metadata.json
models/kmeans.joblib
reports/phase3/clustering_metrics.csv
app/Home.py
app/pages/1_Executive_Dashboard.py
app/pages/2_Customer_Prediction.py
app/pages/3_Batch_Prediction.py
app/pages/4_Customer_Segmentation.py
app/pages/5_Marketing_Analytics.py
```

The app validates the model and data artifacts at startup. If an essential
artifact is missing, it displays a clear error and does not create a substitute.

## 4. Install Python dependencies

Open PowerShell and run:

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional virtual environment

A virtual environment is recommended if this project shares a global Python
installation:

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. Install test dependencies

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pip install -r requirements-dev.txt
```

## 6. Run the application

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m streamlit run app/Home.py
```

Streamlit prints a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser. The pages are available from the sidebar.

### Use a different port

```powershell
python -m streamlit run app/Home.py --server.port 8502
```

### Headless local verification

```powershell
python -m streamlit run app/Home.py --server.headless true --server.port 8505
```

Health endpoint:

```text
http://localhost:8505/_stcore/health
```

Expected response:

```text
ok
```

The live verification for this project returned HTTP 200 for both the health
endpoint and the root page.

## 7. Run automated tests

```powershell
Set-Location "D:\LEARNING\Information System Manangement\Project"
python -m pytest -q
```

Expected Phase 6 result:

```text
57 passed
```

Optional existing smoke checks:

```powershell
python src\phase5_app_checks.py
```

## 8. Configuration

The local Streamlit configuration is `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 50

[browser]
gatherUsageStats = false
```

- Batch uploads are limited to 50 MB.
- Streamlit usage statistics are disabled for this local app.

The batch page also restricts accepted uploads to CSV.

## 9. Environment variables and secrets

No environment variables are needed.

- No `.env.example` is supplied because there are no required environment
  variables.
- No API keys or passwords are used.
- `.gitignore` excludes `.env`, `.env.*`, and `.streamlit/secrets.toml` in case
  future local configuration is added.
- Do not place real credentials in the repository.

## 10. Data and model handling

- The original dataset is `data/raw/bank-full.csv`.
- The app reads the raw dataset and saved artifacts; it does not overwrite them.
- The raw dataset checksum is covered by a pytest integrity check.
- The uploaded batch file is held in the Streamlit session and is not written to
  disk by the application.
- The Random Forest artifact is 44 MB and may take a short time to load on first
  use. Streamlit caching avoids repeated reloads during a session.

## 11. Troubleshooting

### `python` is not recognized

Install Python or use the full path to the Python executable. Confirm with:

```powershell
python --version
```

### `streamlit` is not recognized

Use the module form:

```powershell
python -m streamlit run app/Home.py
```

### Missing artifact error

Verify the required paths from section 3. Do not create a replacement model if a
Phase 3/4 artifact is missing; restore the original artifact or complete the
missing phase properly.

### Port already in use

Use another port:

```powershell
python -m streamlit run app/Home.py --server.port 8502
```

### CSV rejected

Check required columns, data types, binary values, ranges, and blank categories
against the User Manual.

### The first page is slow

The initial load reads the dataset and the large model artifact. Subsequent cached
use within the same session is faster.

## 12. Local deployment boundary

This guide prepares a local Streamlit deployment only. It does not include:

- Authentication or role-based access.
- HTTPS/TLS termination.
- Cloud hosting.
- External database or object storage.
- Production monitoring or alerting.
- Privacy review for real customer data.

Those items belong to future production work, not Phase 6.

Phase 7 was not started.
