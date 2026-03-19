from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from churn.config import ProfitParams
from churn.explain.shap_utils import shap_for_rows, top_local_contributors
from churn.model.predict import decide, load_artifacts, optimal_threshold_from_metadata, predict_proba
from churn.recommend.engine import recommend_retention_actions


app = FastAPI(title="Churn Management API", version="0.1.0")

ART = None
WEB_DIR = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    customer_id: str = Field(default="C000000")
    age: int | None = None
    tenure_months: int | None = None
    contract_type: str | None = None
    payment_method: str | None = None
    paperless_billing: str | None = None
    international_plan: str | None = None
    avg_monthly_gb: float | None = None
    support_tickets_6m: int | None = None
    outages_3m: int | None = None
    late_payments_12m: int | None = None
    app_sessions_30d: int | None = None
    add_ons: int | None = None
    monthly_charges: float | None = None
    total_charges: float | None = None


class PredictRequest(BaseModel):
    customers: list[CustomerFeatures]
    retention_gain: float = 120.0
    retention_cost: float = 20.0
    threshold: float | None = None
    include_shap: bool = True
    include_recommendations: bool = True


@app.on_event("startup")
def _startup() -> None:
    global ART
    ART = load_artifacts()

    # Serve the HTML UI if present
    global WEB_DIR
    try:
        from pathlib import Path

        # src/churn/api/main.py -> api -> churn -> src -> project root
        WEB_DIR = Path(__file__).resolve().parents[3] / "web"
        if WEB_DIR.exists():
            app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    except Exception:
        WEB_DIR = None


@app.get("/ui")
def ui() -> Any:
    if WEB_DIR is None:
        return {"error": "UI not found. Ensure `web/index.html` exists."}
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": ART is not None}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    assert ART is not None

    df = pd.DataFrame([c.model_dump() for c in req.customers])
    proba = predict_proba(df, ART)

    profit = ProfitParams(retention_gain=req.retention_gain, retention_cost=req.retention_cost)
    threshold = float(req.threshold) if req.threshold is not None else optimal_threshold_from_metadata(ART, profit)
    pred = decide(proba, threshold)

    out = []

    shap_payload = None
    if req.include_shap:
        feature_names_out = ART.metadata.get("feature_names_out", None)
        sr = shap_for_rows(df, ART.preprocessor, ART.model, feature_names_out=feature_names_out)
        shap_payload = [top_local_contributors(sr, i, top_k=8) for i in range(len(df))]

    for i in range(len(df)):
        recs = []
        if req.include_recommendations:
            recs = [r.__dict__ for r in recommend_retention_actions(df.iloc[i])]

        out.append(
            {
                "customer_id": df.iloc[i].get("customer_id"),
                "churn_probability": float(proba[i]),
                "threshold": float(threshold),
                "will_churn": int(pred[i]),
                "top_shap": shap_payload[i] if shap_payload is not None else None,
                "recommendations": recs,
            }
        )

    return {"results": out}


def _run_dev() -> None:
    """
    Allows running with: `python src/churn/api/main.py`
    (useful on devices where uvicorn command isn't available on PATH).
    """
    import uvicorn

    uvicorn.run("churn.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    _run_dev()

