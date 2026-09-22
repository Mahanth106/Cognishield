# CogniShield

CogniShield is a synthetic insider-threat detection prototype. It combines user activity telemetry, behavioral anomaly models, lexical indicators, explainable scoring, SQLite persistence, a FastAPI service, and a Streamlit SOC dashboard.

## Features

- Synthetic logon, device, file, and text telemetry generation
- Daily user feature extraction
- Autoencoder, LSTM, and Isolation Forest anomaly scoring
- Keyword-based lexical risk detection
- Composite Low, Medium, and High risk classification
- SHAP and reconstruction-error attribution data
- SQLite persistence with CSV exports
- FastAPI endpoints with Swagger documentation
- Streamlit security operations dashboard

## Setup

Create or activate the project virtual environment, then install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

Run these commands from the project root:

```powershell
python -m src.data_generator
python -m src.feature_extraction
python run_pipeline.py
```

The generator uses a fixed seed for a reproducible demonstration dataset. It creates one confirmed exfiltration scenario and four secondary behavioral scenarios.

## Start the applications

Start the API:

```powershell
python -m uvicorn src.api:app --reload
```

Open Swagger at `http://127.0.0.1:8000/docs`.

Start the dashboard in a second terminal:

```powershell
streamlit run dashboard/app.py
```

The dashboard is normally available at `http://localhost:8501/`.

## Current scope

This is a proof-of-concept using synthetic data and batch scoring. The mitigation endpoint validates and records simulation requests; it does not change accounts or access controls. Authentication, authorization, production model evaluation, and encrypted database storage are not included yet.