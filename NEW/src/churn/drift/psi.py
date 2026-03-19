from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PSIMetric:
    feature: str
    psi: float


def _psi_for_series(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = expected.dropna()
    a = actual.dropna()
    if len(e) < 50 or len(a) < 50:
        return float("nan")

    # If categorical: use value counts
    if e.dtype == "object" or a.dtype == "object":
        e_dist = e.value_counts(normalize=True)
        a_dist = a.value_counts(normalize=True)
        keys = sorted(set(e_dist.index).union(set(a_dist.index)))
        eps = 1e-6
        psi = 0.0
        for k in keys:
            ee = float(e_dist.get(k, 0.0)) + eps
            aa = float(a_dist.get(k, 0.0)) + eps
            psi += (aa - ee) * np.log(aa / ee)
        return float(psi)

    # Numeric: bin by quantiles from expected
    try:
        quantiles = np.unique(np.quantile(e.to_numpy(), np.linspace(0, 1, bins + 1)))
        if len(quantiles) <= 2:
            return 0.0
        e_bins = pd.cut(e, bins=quantiles, include_lowest=True)
        a_bins = pd.cut(a, bins=quantiles, include_lowest=True)
    except Exception:
        return float("nan")

    e_dist = e_bins.value_counts(normalize=True)
    a_dist = a_bins.value_counts(normalize=True)
    eps = 1e-6
    psi = 0.0
    for b in e_dist.index:
        ee = float(e_dist.get(b, 0.0)) + eps
        aa = float(a_dist.get(b, 0.0)) + eps
        psi += (aa - ee) * np.log(aa / ee)
    return float(psi)


def psi_report(
    baseline_df: pd.DataFrame,
    new_df: pd.DataFrame,
    features: list[str],
    bins: int = 10,
) -> list[PSIMetric]:
    out: list[PSIMetric] = []
    for f in features:
        if f not in baseline_df.columns or f not in new_df.columns:
            continue
        out.append(PSIMetric(feature=f, psi=_psi_for_series(baseline_df[f], new_df[f], bins=bins)))
    out.sort(key=lambda x: (np.nan_to_num(x.psi, nan=-1.0)), reverse=True)
    return out


def drift_level(psi_value: float) -> str:
    # Common interpretation thresholds
    if np.isnan(psi_value):
        return "unknown"
    if psi_value < 0.1:
        return "low"
    if psi_value < 0.25:
        return "medium"
    return "high"


def serialize_baseline(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"features": features, "sample_size": int(len(df))}
    for f in features:
        if f not in df.columns:
            continue
        s = df[f]
        if s.dtype == "object":
            payload[f] = {"type": "categorical", "dist": s.value_counts(normalize=True).to_dict()}
        else:
            s2 = pd.to_numeric(s, errors="coerce").dropna()
            payload[f] = {
                "type": "numeric",
                "quantiles": [float(x) for x in np.unique(np.quantile(s2.to_numpy(), np.linspace(0, 1, 11)))],
            }
    return payload

