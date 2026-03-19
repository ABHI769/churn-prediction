from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from churn.features.preprocess import add_engineered_features


@dataclass(frozen=True)
class ShapResult:
    feature_names: list[str]
    shap_values: np.ndarray
    base_values: np.ndarray


def make_tree_explainer(model: Any) -> shap.TreeExplainer:
    # For XGBoost, TreeExplainer is fast and accurate
    return shap.TreeExplainer(model)


def shap_for_rows(
    df_rows: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    feature_names_out: list[str] | None = None,
) -> ShapResult:
    X = add_engineered_features(df_rows)
    X = X.drop(columns=["churn"], errors="ignore")
    X_t = preprocessor.transform(X)

    explainer = make_tree_explainer(model)
    sv = explainer.shap_values(X_t)
    base = explainer.expected_value

    shap_values = np.asarray(sv)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    base_values = np.asarray(base)
    if base_values.ndim == 0:
        base_values = np.full((shap_values.shape[0],), float(base_values))

    if feature_names_out is None:
        try:
            feature_names_out = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names_out = [f"f{i}" for i in range(shap_values.shape[1])]

    return ShapResult(feature_names=feature_names_out, shap_values=shap_values, base_values=base_values)


def top_local_contributors(
    shap_result: ShapResult,
    row_index: int = 0,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    vals = shap_result.shap_values[row_index]
    idx = np.argsort(np.abs(vals))[::-1][:top_k]
    out = []
    for i in idx:
        out.append(
            {
                "feature": shap_result.feature_names[int(i)],
                "shap": float(vals[int(i)]),
                "direction": "increases_churn" if vals[int(i)] > 0 else "decreases_churn",
            }
        )
    return out

