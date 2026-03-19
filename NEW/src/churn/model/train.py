from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from churn.config import ProfitParams, get_paths
from churn.drift.psi import serialize_baseline
from churn.features.preprocess import FeatureSpec, get_feature_names, make_preprocessor, split_xy
from churn.profit.threshold import find_best_threshold, profit_from_preds
from churn.utils.io import dump_joblib, dump_json, ensure_dir


@dataclass(frozen=True)
class TrainArtifacts:
    model_path: Path
    preprocessor_path: Path
    metadata_path: Path
    baseline_path: Path


def train_xgboost(
    df: pd.DataFrame,
    profit: ProfitParams = ProfitParams(),
    use_smote: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    spec = FeatureSpec()
    X_df, y, ids = split_xy(df, spec)

    X_train_df, X_test_df, y_train, y_test, ids_train, ids_test = train_test_split(
        X_df,
        y,
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pre, spec2 = make_preprocessor(spec)
    X_train = pre.fit_transform(X_train_df)
    X_test = pre.transform(X_test_df)

    # Optional imbalance handling: SMOTE in transformed space
    if use_smote:
        sm = SMOTE(random_state=random_state, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # Class weighting also helps when SMOTE is off
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1.0)) if not use_smote else 1.0

    model = XGBClassifier(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=0,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    default_pred = (proba >= 0.5).astype(int)
    default_profit = float(profit_from_preds(y_test, default_pred, profit.retention_gain, profit.retention_cost))

    best_t, best_p = find_best_threshold(
        y_test,
        proba,
        retention_gain=profit.retention_gain,
        retention_cost=profit.retention_cost,
    )
    best_pred = (proba >= best_t).astype(int)

    metrics = {
        "roc_auc": auc,
        "accuracy_default_0.5": float(accuracy_score(y_test, default_pred)),
        "confusion_matrix_default_0.5": confusion_matrix(y_test, default_pred).tolist(),
        "profit_default_0.5": default_profit,
        "best_threshold_profit": best_t,
        "profit_best_threshold": float(best_p),
        "accuracy_best_threshold": float(accuracy_score(y_test, best_pred)),
        "confusion_matrix_best_threshold": confusion_matrix(y_test, best_pred).tolist(),
        "classification_report_best_threshold": classification_report(y_test, best_pred, output_dict=True),
    }

    feature_names = get_feature_names(pre)
    baseline_payload = serialize_baseline(
        df=df.drop(columns=["churn"], errors="ignore"),
        features=[c for c in df.columns if c not in ["churn"]],
    )

    payload = {
        "metrics": metrics,
        "profit_params": {"retention_gain": profit.retention_gain, "retention_cost": profit.retention_cost},
        "feature_names_out": feature_names,
        "spec": {
            "id_col": spec2.id_col,
            "target_col": spec2.target_col,
            "numeric": list(spec2.numeric),
            "categorical": list(spec2.categorical),
        },
    }

    return {
        "model": model,
        "preprocessor": pre,
        "metadata": payload,
        "baseline": baseline_payload,
        "holdout": {
            "X_test_df": X_test_df.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "ids_test": ids_test.reset_index(drop=True),
            "proba": pd.Series(proba),
        },
    }


def save_artifacts(bundle: dict[str, Any], out_dir: Path | None = None) -> TrainArtifacts:
    paths = get_paths()
    artifacts = out_dir or paths.artifacts_dir
    ensure_dir(artifacts)

    model_path = artifacts / "xgb_model.joblib"
    pre_path = artifacts / "preprocessor.joblib"
    metadata_path = artifacts / "metadata.json"
    baseline_path = artifacts / "baseline_schema.json"

    dump_joblib(bundle["model"], model_path)
    dump_joblib(bundle["preprocessor"], pre_path)
    dump_json(bundle["metadata"], metadata_path)
    dump_json(bundle["baseline"], baseline_path)

    return TrainArtifacts(
        model_path=model_path,
        preprocessor_path=pre_path,
        metadata_path=metadata_path,
        baseline_path=baseline_path,
    )

