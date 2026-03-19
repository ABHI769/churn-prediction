from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from churn.config import ProfitParams, get_paths
from churn.data.synthetic import SyntheticChurnConfig, make_synthetic_churn_df
from churn.drift.psi import drift_level, psi_report
from churn.explain.shap_utils import shap_for_rows, top_local_contributors
from churn.model.predict import decide, load_artifacts, optimal_threshold_from_metadata, predict_proba
from churn.recommend.engine import recommend_retention_actions
from churn.utils.io import load_json


st.set_page_config(page_title="Churn Management Dashboard", layout="wide")

st.title("Churn Management System")
st.caption("Prediction • Profit optimization • SHAP explainability • Recommendations • Drift monitoring (PSI)")


paths = get_paths()
artifacts_dir = paths.artifacts_dir

left, right = st.columns([0.35, 0.65], gap="large")

with left:
    st.subheader("Inputs")
    mode = st.radio("Data source", options=["Synthetic (demo)", "Upload CSV"], index=0)
    n_rows = st.slider("Synthetic rows", 1000, 30000, 8000, step=1000)
    retention_gain = st.number_input("Retention gain (profit per saved churner)", value=120.0, min_value=0.0)
    retention_cost = st.number_input("Retention cost (cost per contacted non-churn)", value=20.0, min_value=0.0)
    show_shap = st.checkbox("Show SHAP explanations", value=True)

    uploaded = None
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV with churn features", type=["csv"])

with right:
    st.subheader("Model status")
    if not artifacts_dir.exists():
        st.warning("No artifacts found. Run training: `python -m scripts.train`")
        st.stop()

    try:
        art = load_artifacts()
        meta = art.metadata
        st.success("Model artifacts loaded.")
        st.json(
            {
                "roc_auc": meta.get("metrics", {}).get("roc_auc"),
                "best_threshold_profit": meta.get("metrics", {}).get("best_threshold_profit"),
                "profit_best_threshold": meta.get("metrics", {}).get("profit_best_threshold"),
            }
        )
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()


if mode == "Upload CSV" and uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = make_synthetic_churn_df(SyntheticChurnConfig(n_rows=n_rows))

st.divider()

st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

df_features = df.drop(columns=["churn"], errors="ignore")
proba = predict_proba(df_features, art)

profit = ProfitParams(retention_gain=retention_gain, retention_cost=retention_cost)
threshold = optimal_threshold_from_metadata(art, profit)
pred = decide(proba, threshold)

st.subheader("Churn risk distribution")
fig = px.histogram(pd.DataFrame({"churn_probability": proba}), x="churn_probability", nbins=30)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top at-risk customers")
top_n = st.slider("Show top N", 10, 200, 30)
ranked = df_features.copy()
ranked["churn_probability"] = proba
ranked["will_churn"] = pred
ranked = ranked.sort_values("churn_probability", ascending=False).head(int(top_n))
st.dataframe(ranked, use_container_width=True)

st.subheader("Customer drill-down")
row_idx = st.number_input("Row index", min_value=0, max_value=len(df_features) - 1, value=0, step=1)
row = df_features.iloc[int(row_idx)]

colA, colB = st.columns([0.5, 0.5], gap="large")
with colA:
    st.markdown("**Prediction**")
    st.write(
        {
            "customer_id": row.get("customer_id"),
            "churn_probability": float(proba[int(row_idx)]),
            "threshold": float(threshold),
            "will_churn": int(pred[int(row_idx)]),
        }
    )

with colB:
    st.markdown("**Recommendations**")
    recs = recommend_retention_actions(row)
    for r in recs:
        st.write({"action": r.action, "reason": r.reason, "estimated_cost": r.estimated_cost})

if show_shap:
    st.subheader("Local explanation (SHAP)")
    try:
        sr = shap_for_rows(df_features.iloc[[int(row_idx)]], art.preprocessor, art.model, meta.get("feature_names_out"))
        st.dataframe(pd.DataFrame(top_local_contributors(sr, 0, top_k=12)), use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")

st.subheader("Drift monitoring (PSI)")
try:
    baseline_path = artifacts_dir / "baseline_schema.json"
    baseline = load_json(baseline_path)
    # Compare current data to a baseline built on training-like columns.
    features = baseline.get("features", [])
    report = psi_report(pd.DataFrame({c: df_features.get(c) for c in features}), df_features, features=features)
    rep_df = pd.DataFrame([{"feature": r.feature, "psi": r.psi, "level": drift_level(r.psi)} for r in report]).head(25)
    st.dataframe(rep_df, use_container_width=True)
except Exception as e:
    st.warning(f"Drift report unavailable: {e}")

