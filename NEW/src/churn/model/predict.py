from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.config import ProfitParams, get_paths
from churn.features.preprocess import FeatureSpec, add_engineered_features
from churn.profit.threshold import find_best_threshold
from churn.utils.io import load_joblib, load_json


@dataclass(frozen=True)
class LoadedArtifacts:
    model: Any
    preprocessor: Any
    metadata: dict[str, Any]


def load_artifacts(artifacts_dir: Path | None = None) -> LoadedArtifacts:
    paths = get_paths()
    d = artifacts_dir or paths.artifacts_dir
    model = load_joblib(d / "xgb_model.joblib")
    pre = load_joblib(d / "preprocessor.joblib")
    meta = load_json(d / "metadata.json")
    return LoadedArtifacts(model=model, preprocessor=pre, metadata=meta)


def predict_proba(df: pd.DataFrame, artifacts: LoadedArtifacts) -> np.ndarray:
    spec = FeatureSpec()
    X = add_engineered_features(df).drop(columns=[spec.target_col], errors="ignore")
    X_t = artifacts.preprocessor.transform(X)
    proba = artifacts.model.predict_proba(X_t)[:, 1]
    return np.asarray(proba, dtype=float)


def optimal_threshold_from_metadata(
    artifacts: LoadedArtifacts,
    profit: ProfitParams,
) -> float:
    # Use stored best threshold if profit params match training; otherwise compute on holdout is not available here.
    train_p = artifacts.metadata.get("profit_params", {})
    if (
        float(train_p.get("retention_gain", profit.retention_gain)) == float(profit.retention_gain)
        and float(train_p.get("retention_cost", profit.retention_cost)) == float(profit.retention_cost)
    ):
        return float(artifacts.metadata["metrics"]["best_threshold_profit"])
    return float(artifacts.metadata["metrics"]["best_threshold_profit"])


def decide(
    proba: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (np.asarray(proba) >= float(threshold)).astype(int)

