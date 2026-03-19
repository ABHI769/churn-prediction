from __future__ import annotations

import numpy as np


def profit_from_preds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    retention_gain: float,
    retention_cost: float,
) -> float:
    """
    Profit = TP * gain - FP * cost
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return tp * float(retention_gain) - fp * float(retention_cost)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    retention_gain: float,
    retention_cost: float,
    grid_size: int = 301,
) -> tuple[float, float]:
    """
    Grid-search a threshold in [0,1] maximizing the profit function.
    Returns (best_threshold, best_profit).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_t = 0.5
    best_p = -np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p = profit_from_preds(y_true, y_pred, retention_gain, retention_cost)
        if p > best_p:
            best_p = p
            best_t = float(t)
    return best_t, float(best_p)

