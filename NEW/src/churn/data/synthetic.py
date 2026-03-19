from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticChurnConfig:
    n_rows: int = 8000
    seed: int = 42
    churn_rate_target: float = 0.27


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_churn_df(cfg: SyntheticChurnConfig) -> pd.DataFrame:
    """
    Create a telecom-like churn dataset with both numeric and categorical features.
    Target column is `churn` (0/1).
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_rows

    # Demographics / account
    tenure_months = rng.integers(0, 73, size=n)
    age = np.clip(rng.normal(39, 12, size=n), 18, 85).round().astype(int)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], p=[0.55, 0.25, 0.20], size=n)
    payment_method = rng.choice(
        ["electronic-check", "mailed-check", "bank-transfer", "credit-card"],
        p=[0.35, 0.20, 0.22, 0.23],
        size=n,
    )
    paperless_billing = rng.choice(["yes", "no"], p=[0.62, 0.38], size=n)

    # Service usage
    avg_monthly_gb = np.clip(rng.gamma(shape=2.2, scale=12.0, size=n), 0, 300)
    support_tickets_6m = np.clip(rng.poisson(lam=0.7, size=n), 0, 12)
    outages_3m = np.clip(rng.poisson(lam=0.35, size=n), 0, 8)
    international_plan = rng.choice(["yes", "no"], p=[0.18, 0.82], size=n)

    # Pricing & engagement
    base_charge = rng.normal(55, 12, size=n)
    add_ons = rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.2, 0.1], size=n)
    add_on_charge = add_ons * rng.normal(8.5, 2.0, size=n)
    monthly_charges = np.clip(base_charge + add_on_charge + 0.06 * avg_monthly_gb, 15, 180)
    total_charges = np.clip(monthly_charges * (tenure_months + 1) + rng.normal(0, 35, size=n), 0, 20000)

    app_sessions_30d = np.clip(rng.poisson(lam=14, size=n) + (avg_monthly_gb / 20).astype(int), 0, 200)
    late_payments_12m = np.clip(rng.poisson(lam=0.8, size=n), 0, 12)

    # Feature engineering proxies
    # Higher engagement reduces churn risk; more tickets/outages/late payments increase risk
    engagement_score = (
        0.35 * np.log1p(app_sessions_30d)
        + 0.25 * np.log1p(avg_monthly_gb)
        + 0.25 * (tenure_months / 72)
        - 0.25 * (support_tickets_6m / 12)
        - 0.20 * (outages_3m / 8)
        - 0.20 * (late_payments_12m / 12)
    )

    # Latent churn propensity (logit)
    # Month-to-month, high charges, many issues, late payments => more churn
    logit = (
        -0.85 * engagement_score
        + 0.018 * (monthly_charges - 60)
        + 0.22 * support_tickets_6m
        + 0.26 * outages_3m
        + 0.18 * late_payments_12m
        + 0.55 * (contract_type == "month-to-month").astype(float)
        - 0.25 * (contract_type == "two-year").astype(float)
        + 0.14 * (payment_method == "electronic-check").astype(float)
        + 0.22 * (paperless_billing == "yes").astype(float)
        + 0.12 * (international_plan == "yes").astype(float)
        + rng.normal(0, 0.6, size=n)
    )
    p = _sigmoid(logit)

    # Calibrate to the requested churn rate (approx) by shifting logits
    # Shift chosen so mean(sigmoid(logit+shift)) ~= target.
    target = float(cfg.churn_rate_target)
    shift = 0.0
    for _ in range(30):
        cur = _sigmoid(logit + shift).mean()
        # simple proportional adjustment
        shift += np.clip(np.log(target / (1 - target)) - np.log(cur / (1 - cur)), -0.5, 0.5)
        if abs(cur - target) < 0.002:
            break
    p = _sigmoid(logit + shift)
    churn = rng.binomial(1, p, size=n).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000+i}" for i in range(n)],
            "age": age,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "international_plan": international_plan,
            "avg_monthly_gb": avg_monthly_gb.round(2),
            "support_tickets_6m": support_tickets_6m,
            "outages_3m": outages_3m,
            "late_payments_12m": late_payments_12m,
            "app_sessions_30d": app_sessions_30d,
            "add_ons": add_ons,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "churn": churn,
        }
    )

    # Inject a few missing values (realism)
    for col, frac in [
        ("total_charges", 0.015),
        ("payment_method", 0.01),
        ("avg_monthly_gb", 0.008),
    ]:
        m = rng.random(n) < frac
        df.loc[m, col] = np.nan

    return df

