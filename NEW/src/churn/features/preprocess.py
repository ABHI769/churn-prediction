from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    id_col: str = "customer_id"
    target_col: str = "churn"

    numeric: tuple[str, ...] = (
        "age",
        "tenure_months",
        "avg_monthly_gb",
        "support_tickets_6m",
        "outages_3m",
        "late_payments_12m",
        "app_sessions_30d",
        "add_ons",
        "monthly_charges",
        "total_charges",
    )

    categorical: tuple[str, ...] = (
        "contract_type",
        "payment_method",
        "paperless_billing",
        "international_plan",
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Non-linear behavior signals
    out["charge_per_gb"] = (out["monthly_charges"] / (out["avg_monthly_gb"].fillna(0) + 1.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["issues_score"] = (
        out["support_tickets_6m"].fillna(0) * 1.0
        + out["outages_3m"].fillna(0) * 1.2
        + out["late_payments_12m"].fillna(0) * 0.9
    )
    out["tenure_bucket"] = pd.cut(
        out["tenure_months"].fillna(0),
        bins=[-1, 6, 12, 24, 48, 72, 10_000],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72", "72+"],
    ).astype(object)

    return out


def make_preprocessor(spec: FeatureSpec) -> tuple[ColumnTransformer, FeatureSpec]:
    # Update feature spec to include engineered columns
    numeric = tuple(spec.numeric) + ("charge_per_gb", "issues_score")
    categorical = tuple(spec.categorical) + ("tenure_bucket",)
    spec2 = FeatureSpec(id_col=spec.id_col, target_col=spec.target_col, numeric=numeric, categorical=categorical)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric)),
            ("cat", cat_pipe, list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return pre, spec2


def split_xy(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df2 = add_engineered_features(df)
    ids = df2[spec.id_col] if spec.id_col in df2.columns else pd.Series([None] * len(df2))
    y = df2[spec.target_col].astype(int)
    X = df2.drop(columns=[c for c in [spec.target_col] if c in df2.columns])
    return X, y, ids


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
    return names

