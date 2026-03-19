from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Recommendation:
    action: str
    reason: str
    estimated_cost: float


def recommend_retention_actions(row: pd.Series) -> list[Recommendation]:
    """
    Rule-based recommendation engine (simple, explainable).
    Expects original (non-one-hot) columns.
    """
    recs: list[Recommendation] = []

    monthly = float(row.get("monthly_charges", 0) or 0)
    tenure = float(row.get("tenure_months", 0) or 0)
    gb = float(row.get("avg_monthly_gb", 0) or 0)
    tickets = float(row.get("support_tickets_6m", 0) or 0)
    outages = float(row.get("outages_3m", 0) or 0)
    late = float(row.get("late_payments_12m", 0) or 0)
    contract = str(row.get("contract_type", "") or "").lower()

    high_value = monthly >= 80 or (monthly >= 65 and tenure >= 24)
    high_usage = gb >= 50
    low_engagement = float(row.get("app_sessions_30d", 0) or 0) <= 6
    many_issues = tickets >= 2 or outages >= 2

    if contract == "month-to-month" and monthly >= 70:
        recs.append(
            Recommendation(
                action="Offer 10–15% discount for 3 months",
                reason="Month-to-month + higher bill increases churn sensitivity.",
                estimated_cost=0.12 * monthly * 3,
            )
        )

    if many_issues:
        recs.append(
            Recommendation(
                action="Proactive support call + service quality check",
                reason="High support tickets/outages are strong churn drivers.",
                estimated_cost=15.0,
            )
        )

    if late >= 2:
        recs.append(
            Recommendation(
                action="Flexible payment plan / autopay incentive",
                reason="Repeated late payments signal friction and risk.",
                estimated_cost=8.0,
            )
        )

    if low_engagement and not many_issues:
        recs.append(
            Recommendation(
                action="Loyalty program + onboarding tips",
                reason="Low engagement suggests customer not seeing value.",
                estimated_cost=10.0,
            )
        )

    if high_value and high_usage:
        recs.append(
            Recommendation(
                action="Free upgrade add-on for 1 month (premium retention)",
                reason="High value/high usage customer; keep experience premium.",
                estimated_cost=12.0,
            )
        )

    if not recs:
        recs.append(
            Recommendation(
                action="Targeted check-in message",
                reason="No strong risk pattern detected; lightweight engagement.",
                estimated_cost=2.0,
            )
        )

    return recs

