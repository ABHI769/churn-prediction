# Churn Management System (End-to-End)

This project is an end-to-end churn prediction + decision-support system:
- Churn probability prediction (XGBoost)
- Imbalance handling (SMOTE or class-weighting)
- Profit-based threshold optimization (maximize business profit, not accuracy)
- Explainable AI (SHAP global + local)
- Recommendation engine (personalized retention actions)
- Drift detection (PSI) to monitor concept drift
- FastAPI service for real-time inference
- Streamlit dashboard for interactive analysis

## Quickstart (Windows PowerShell)

Create a virtualenv (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train a model (also generates synthetic data if you don’t provide any):

```bash
python run_train.py --n-rows 8000
```

Run the API:

```bash
python run_api.py
```

Open the UI:

- `http://127.0.0.1:8000/` (served from `web/index.html`)

Alternative (double click):
- `start_api.ps1` (PowerShell)
- `start_api.bat` (CMD)

Run the dashboard:

```bash
$env:PYTHONPATH="$pwd\\src"
streamlit run src/churn/dashboard/app.py
```

## Project structure

- `src/churn/data/synthetic.py`: synthetic churn dataset generator
- `src/churn/features/preprocess.py`: preprocessing + feature engineering
- `src/churn/model/train.py`: training + evaluation + artifact saving
- `src/churn/model/predict.py`: inference wrapper
- `src/churn/profit/threshold.py`: profit function + best threshold search
- `src/churn/explain/shap_utils.py`: SHAP global/local explanations
- `src/churn/recommend/engine.py`: retention recommendation logic
- `src/churn/drift/psi.py`: drift detection utilities (PSI)
- `src/churn/api/main.py`: FastAPI endpoints
- `src/churn/dashboard/app.py`: Streamlit dashboard
- `artifacts/`: saved model + preprocessors + baselines

## Notes

- Profit parameters (retention gain/cost) are configurable at inference time in the API and dashboard.
- Drift is monitored using PSI between a stored baseline distribution (from training data) and new incoming data.

