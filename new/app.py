from datetime import date, datetime, timedelta
import os
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

app = Flask(__name__)

rf_model = None
gb_model = None
feature_columns = None
analytics_cache = {}
training_summary = {}

NUMERIC_COLUMNS = [
    "age",
    "tenure",
    "income",
    "dataUsage",
    "minutesUsed",
    "smsSent",
    "avgCallDuration",
    "serviceCalls",
    "monthlyCharges",
    "totalCharges",
    "outstandingBalance",
    "emailOpenRate",
    "satisfactionScore",
    "nps",
    "numComplaints",
    "supportTickets",
    "daysSinceActivity",
    "inactivityPeriod",
    "daysSinceLastPayment",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "occupation",
    "location",
    "segment",
    "multipleLines",
    "internet",
    "onlineSecurity",
    "techSupport",
    "streamingTV",
    "streamingMovies",
    "deviceProtection",
    "premiumSupport",
    "lastLogin",
    "usageTrend",
    "contract",
    "payment",
    "paymentHistory",
    "autoPay",
    "offerResponse",
    "loyaltyMember",
    "planDowngrade",
    "competitorInteraction",
]

BOOLEAN_COLUMNS = [
    "senior",
    "partner",
    "dependents",
    "phone",
    "paperless",
]

ALL_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS

BASELINE_PROFILE = {
    "age": 38,
    "tenure": 24,
    "income": 65000,
    "dataUsage": 48,
    "minutesUsed": 380,
    "smsSent": 90,
    "avgCallDuration": 4.5,
    "serviceCalls": 1,
    "monthlyCharges": 2800,
    "totalCharges": 67200,
    "outstandingBalance": 0,
    "emailOpenRate": 42,
    "satisfactionScore": 4,
    "nps": 8,
    "numComplaints": 0,
    "supportTickets": 1,
    "daysSinceActivity": 4,
    "inactivityPeriod": 0,
    "daysSinceLastPayment": 10,
    "gender": "male",
    "occupation": "salaried",
    "location": "urban",
    "segment": "standard",
    "multipleLines": "no",
    "internet": "fiber",
    "onlineSecurity": "yes",
    "techSupport": "yes",
    "streamingTV": "yes",
    "streamingMovies": "yes",
    "deviceProtection": "yes",
    "premiumSupport": "no",
    "lastLogin": "3_days",
    "usageTrend": "stable",
    "contract": "yearly",
    "payment": "credit",
    "paymentHistory": "no_missed",
    "autoPay": "yes",
    "offerResponse": "responded",
    "loyaltyMember": "yes",
    "planDowngrade": "no",
    "competitorInteraction": "no",
    "senior": False,
    "partner": True,
    "dependents": False,
    "phone": True,
    "paperless": True,
}


def clamp(value, low, high):
    return max(low, min(high, value))


def bool_from_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_days_since_payment(value):
    if not value:
        return 15
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
        return clamp((date.today() - parsed).days, 0, 365)
    except ValueError:
        return 15


def build_customer_payload(data):
    payload = {
        "gender": data.get("gender", "male"),
        "senior": bool_from_value(data.get("senior", False)),
        "partner": bool_from_value(data.get("partner", False)),
        "dependents": bool_from_value(data.get("dependents", False)),
        "age": int(float(data.get("age", 35))),
        "tenure": int(float(data.get("tenure", 12))),
        "occupation": data.get("occupation", "salaried"),
        "income": float(data.get("income", 60000)),
        "location": data.get("location", "urban"),
        "segment": data.get("segment", "standard"),
        "phone": bool_from_value(data.get("phone", True)),
        "multipleLines": data.get("multipleLines", "no"),
        "internet": data.get("internet", "fiber"),
        "onlineSecurity": data.get("onlineSecurity", "yes"),
        "techSupport": data.get("techSupport", "no"),
        "streamingTV": data.get("streamingTV", "yes"),
        "streamingMovies": data.get("streamingMovies", "no"),
        "deviceProtection": data.get("deviceProtection", "no"),
        "premiumSupport": data.get("premiumSupport", "no"),
        "dataUsage": float(data.get("dataUsage", 45)),
        "minutesUsed": float(data.get("minutesUsed", 320)),
        "smsSent": float(data.get("smsSent", 120)),
        "avgCallDuration": float(data.get("avgCallDuration", 3.5)),
        "serviceCalls": float(data.get("serviceCalls", 2)),
        "lastLogin": data.get("lastLogin", "3_days"),
        "usageTrend": data.get("usageTrend", "decreasing"),
        "contract": data.get("contract", "monthly"),
        "payment": data.get("payment", "electronic"),
        "paperless": bool_from_value(data.get("paperless", True)),
        "monthlyCharges": float(data.get("monthlyCharges", 3000)),
        "totalCharges": float(data.get("totalCharges", 30000)),
        "paymentHistory": data.get("paymentHistory", "no_missed"),
        "autoPay": data.get("autoPay", "no"),
        "lastPaymentDate": data.get("lastPaymentDate", ""),
        "outstandingBalance": float(data.get("outstandingBalance", 0)),
        "emailOpenRate": float(data.get("emailOpenRate", 45)),
        "offerResponse": data.get("offerResponse", "responded"),
        "satisfactionScore": int(float(data.get("satisfactionScore", 3))),
        "nps": int(float(data.get("nps", 7))),
        "numComplaints": float(data.get("numComplaints", 1)),
        "supportTickets": float(data.get("supportTickets", 1)),
        "loyaltyMember": data.get("loyaltyMember", "no"),
        "daysSinceActivity": float(data.get("daysSinceActivity", 3)),
        "planDowngrade": data.get("planDowngrade", "no"),
        "inactivityPeriod": float(data.get("inactivityPeriod", 0)),
        "competitorInteraction": data.get("competitorInteraction", "no"),
    }
    payload["daysSinceLastPayment"] = parse_days_since_payment(payload["lastPaymentDate"])
    payload["totalCharges"] = max(payload["totalCharges"], payload["monthlyCharges"] * max(payload["tenure"], 1) * 0.4)
    payload["outstandingBalance"] = max(payload["outstandingBalance"], 0)
    payload["dataUsage"] = max(payload["dataUsage"], 0)
    payload["minutesUsed"] = max(payload["minutesUsed"], 0)
    payload["smsSent"] = max(payload["smsSent"], 0)
    payload["daysSinceActivity"] = max(payload["daysSinceActivity"], 0)
    payload["inactivityPeriod"] = max(payload["inactivityPeriod"], 0)
    return payload


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS,
            ),
        ]
    )


def generate_synthetic_dataset(n_samples=12000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 76, n_samples)
    tenure = rng.integers(1, 73, n_samples)
    occupation = rng.choice(
        ["salaried", "self_employed", "retired", "student", "unemployed"],
        size=n_samples,
        p=[0.48, 0.18, 0.11, 0.13, 0.10],
    )
    location = rng.choice(["urban", "semi_urban", "rural"], size=n_samples, p=[0.52, 0.28, 0.20])
    segment = rng.choice(["standard", "premium", "vip"], size=n_samples, p=[0.62, 0.28, 0.10])
    gender = rng.choice(["male", "female"], size=n_samples)

    senior = age >= 60
    partner = rng.random(n_samples) < np.where(age > 28, 0.58, 0.32)
    dependents = rng.random(n_samples) < np.where((age > 30) & partner, 0.46, 0.14)

    income_base = {
        "student": 18000,
        "unemployed": 12000,
        "retired": 38000,
        "self_employed": 72000,
        "salaried": 58000,
    }
    income = np.array([income_base[o] for o in occupation], dtype=float)
    income += (age - 30) * 850 + rng.normal(0, 12000, n_samples)
    income += np.where(segment == "premium", 18000, 0) + np.where(segment == "vip", 45000, 0)
    income = np.clip(income, 8000, 300000)

    phone = rng.random(n_samples) < 0.94
    multiple_lines = np.where(phone & (rng.random(n_samples) < 0.42), "yes", "no")

    internet = rng.choice(["fiber", "dsl", "no"], size=n_samples, p=[0.48, 0.37, 0.15])
    online_security = np.where(
        internet == "no",
        "no",
        np.where(rng.random(n_samples) < np.where(segment == "vip", 0.72, 0.42), "yes", "no"),
    )
    tech_support = np.where(
        internet == "no",
        "no",
        np.where(rng.random(n_samples) < np.where(segment != "standard", 0.58, 0.33), "yes", "no"),
    )
    streaming_tv = np.where(internet == "no", "no", np.where(rng.random(n_samples) < 0.52, "yes", "no"))
    streaming_movies = np.where(internet == "no", "no", np.where(rng.random(n_samples) < 0.49, "yes", "no"))
    device_protection = np.where(internet == "no", "no", np.where(rng.random(n_samples) < 0.38, "yes", "no"))
    premium_support = np.where(rng.random(n_samples) < np.where(segment == "vip", 0.55, 0.14), "yes", "no")

    data_usage = np.where(
        internet == "no",
        rng.normal(1, 1, n_samples),
        np.where(internet == "fiber", rng.normal(65, 22, n_samples), rng.normal(28, 10, n_samples)),
    )
    data_usage = np.clip(data_usage, 0, 250)
    minutes_used = np.clip(rng.normal(360, 130, n_samples) + np.where(multiple_lines == "yes", 70, 0), 20, 1500)
    sms_sent = np.clip(rng.normal(85, 55, n_samples) + np.where(age < 30, 30, -10), 0, 400)
    avg_call_duration = np.clip(rng.normal(4.2, 1.2, n_samples), 0.5, 18)
    service_calls = np.clip(rng.poisson(1.2, n_samples) + np.where(internet == "fiber", 1, 0), 0, 12)

    last_login = rng.choice(["3_days", "1_week", "2_weeks", "1_month"], size=n_samples, p=[0.40, 0.30, 0.18, 0.12])
    usage_trend = rng.choice(["increasing", "stable", "decreasing"], size=n_samples, p=[0.22, 0.53, 0.25])

    contract = rng.choice(["monthly", "yearly", "twoyear"], size=n_samples, p=[0.54, 0.28, 0.18])
    payment = rng.choice(["electronic", "mailed", "bank", "credit"], size=n_samples, p=[0.36, 0.17, 0.22, 0.25])
    paperless = rng.random(n_samples) < 0.68
    payment_history = rng.choice(["no_missed", "1_2_missed", "gt_2_missed"], size=n_samples, p=[0.71, 0.20, 0.09])
    auto_pay = np.where(np.isin(payment, ["bank", "credit"]) & (rng.random(n_samples) < 0.74), "yes", "no")

    monthly_charges = (
        320
        + np.where(phone, 260, 0)
        + np.where(multiple_lines == "yes", 210, 0)
        + np.where(internet == "fiber", 1750, np.where(internet == "dsl", 980, 0))
        + np.where(online_security == "yes", 190, 0)
        + np.where(tech_support == "yes", 240, 0)
        + np.where(streaming_tv == "yes", 220, 0)
        + np.where(streaming_movies == "yes", 220, 0)
        + np.where(device_protection == "yes", 160, 0)
        + np.where(premium_support == "yes", 360, 0)
        + np.where(segment == "premium", 230, 0)
        + np.where(segment == "vip", 620, 0)
        + rng.normal(0, 180, n_samples)
    )
    monthly_charges = np.clip(monthly_charges, 300, 15000)

    total_charges = np.clip(monthly_charges * tenure * rng.uniform(0.78, 1.05, n_samples), 500, 250000)
    outstanding_balance = np.where(
        payment_history == "gt_2_missed",
        rng.normal(4500, 2200, n_samples),
        np.where(payment_history == "1_2_missed", rng.normal(1200, 900, n_samples), rng.normal(120, 250, n_samples)),
    )
    outstanding_balance = np.clip(outstanding_balance, 0, 40000)

    email_open_rate = np.clip(
        rng.normal(46, 16, n_samples)
        + np.where(segment == "vip", 8, 0)
        + np.where(loyalty := rng.random(n_samples) < np.where(segment != "standard", 0.58, 0.22), 7, -4),
        0,
        100,
    )

    offer_response = rng.choice(["responded", "ignored", "unsubscribed"], size=n_samples, p=[0.35, 0.52, 0.13])
    satisfaction_score = np.clip(
        np.round(
            3.6
            + np.where(service_calls >= 4, -1.0, 0)
            + np.where(payment_history == "gt_2_missed", -0.7, 0)
            + np.where(usage_trend == "decreasing", -0.5, 0)
            + np.where(tech_support == "yes", 0.4, 0)
            + rng.normal(0, 0.9, n_samples)
        ),
        1,
        5,
    ).astype(int)

    nps = np.clip(
        np.round(
            6.8
            + (satisfaction_score - 3) * 1.4
            + np.where(offer_response == "responded", 0.6, 0)
            + np.where(offer_response == "unsubscribed", -1.4, 0)
            + rng.normal(0, 1.8, n_samples)
        ),
        0,
        10,
    ).astype(int)

    num_complaints = np.clip(rng.poisson(0.6, n_samples) + np.where(service_calls >= 4, 1, 0), 0, 8)
    support_tickets = np.clip(rng.poisson(1.1, n_samples) + np.where(internet == "fiber", 1, 0), 0, 12)
    loyalty_member = np.where(loyalty, "yes", "no")

    days_since_activity = np.clip(
        np.where(last_login == "3_days", rng.integers(0, 5, n_samples), 0)
        + np.where(last_login == "1_week", rng.integers(5, 10, n_samples), 0)
        + np.where(last_login == "2_weeks", rng.integers(10, 21, n_samples), 0)
        + np.where(last_login == "1_month", rng.integers(21, 45, n_samples), 0),
        0,
        90,
    )
    plan_downgrade = np.where(rng.random(n_samples) < np.where(usage_trend == "decreasing", 0.24, 0.07), "yes", "no")
    inactivity_period = np.clip(days_since_activity + np.where(usage_trend == "decreasing", rng.integers(0, 18, n_samples), 0), 0, 120)
    competitor_interaction = np.where(rng.random(n_samples) < np.where(contract == "monthly", 0.19, 0.07), "yes", "no")
    days_since_last_payment = np.clip(
        np.where(payment_history == "no_missed", rng.integers(0, 21, n_samples), 0)
        + np.where(payment_history == "1_2_missed", rng.integers(10, 45, n_samples), 0)
        + np.where(payment_history == "gt_2_missed", rng.integers(25, 90, n_samples), 0),
        0,
        120,
    )

    df = pd.DataFrame(
        {
            "gender": gender,
            "senior": senior.astype(bool),
            "partner": partner.astype(bool),
            "dependents": dependents.astype(bool),
            "age": age,
            "tenure": tenure,
            "occupation": occupation,
            "income": income,
            "location": location,
            "segment": segment,
            "phone": phone.astype(bool),
            "multipleLines": multiple_lines,
            "internet": internet,
            "onlineSecurity": online_security,
            "techSupport": tech_support,
            "streamingTV": streaming_tv,
            "streamingMovies": streaming_movies,
            "deviceProtection": device_protection,
            "premiumSupport": premium_support,
            "dataUsage": data_usage,
            "minutesUsed": minutes_used,
            "smsSent": sms_sent,
            "avgCallDuration": avg_call_duration,
            "serviceCalls": service_calls,
            "lastLogin": last_login,
            "usageTrend": usage_trend,
            "contract": contract,
            "payment": payment,
            "paperless": paperless.astype(bool),
            "monthlyCharges": monthly_charges,
            "totalCharges": total_charges,
            "paymentHistory": payment_history,
            "autoPay": auto_pay,
            "outstandingBalance": outstanding_balance,
            "emailOpenRate": email_open_rate,
            "offerResponse": offer_response,
            "satisfactionScore": satisfaction_score,
            "nps": nps,
            "numComplaints": num_complaints,
            "supportTickets": support_tickets,
            "loyaltyMember": loyalty_member,
            "daysSinceActivity": days_since_activity,
            "planDowngrade": plan_downgrade,
            "inactivityPeriod": inactivity_period,
            "competitorInteraction": competitor_interaction,
            "daysSinceLastPayment": days_since_last_payment,
        }
    )

    logit = -2.2
    logit += np.where(df["contract"] == "monthly", 1.2, 0)
    logit += np.where(df["contract"] == "yearly", 0.2, 0)
    logit += np.where(df["internet"] == "fiber", 0.55, 0)
    logit += np.where(df["internet"] == "no", -0.65, 0)
    logit += np.where(df["payment"] == "electronic", 0.42, 0)
    logit += np.where(df["autoPay"] == "yes", -0.48, 0)
    logit += np.where(df["paymentHistory"] == "1_2_missed", 0.72, 0)
    logit += np.where(df["paymentHistory"] == "gt_2_missed", 1.25, 0)
    logit += np.where(df["usageTrend"] == "decreasing", 0.9, 0)
    logit += np.where(df["lastLogin"] == "1_month", 1.0, 0)
    logit += np.where(df["lastLogin"] == "2_weeks", 0.45, 0)
    logit += np.where(df["competitorInteraction"] == "yes", 1.0, 0)
    logit += np.where(df["planDowngrade"] == "yes", 0.95, 0)
    logit += np.where(df["techSupport"] == "yes", -0.42, 0)
    logit += np.where(df["onlineSecurity"] == "yes", -0.24, 0)
    logit += np.where(df["premiumSupport"] == "yes", -0.22, 0)
    logit += np.where(df["loyaltyMember"] == "yes", -0.3, 0)
    logit += np.where(df["offerResponse"] == "responded", -0.25, 0)
    logit += np.where(df["offerResponse"] == "unsubscribed", 0.7, 0)
    logit += np.where(df["segment"] == "vip", -0.3, 0)
    logit += np.where(df["tenure"] < 6, 0.8, 0)
    logit += np.where(df["tenure"] < 12, 0.35, 0)
    logit += np.where(df["daysSinceActivity"] > 21, 0.7, 0)
    logit += np.where(df["inactivityPeriod"] > 14, 0.55, 0)
    logit += np.where(df["serviceCalls"] >= 4, 0.65, 0)
    logit += np.where(df["numComplaints"] >= 2, 0.55, 0)
    logit += np.where(df["supportTickets"] >= 3, 0.35, 0)
    logit += np.where(df["satisfactionScore"] <= 2, 1.0, 0)
    logit += np.where(df["nps"] <= 4, 0.7, 0)
    logit += np.where(df["outstandingBalance"] > 1500, 0.55, 0)
    logit += np.where(df["monthlyCharges"] > 4500, 0.25, 0)
    logit += np.where(df["emailOpenRate"] < 15, 0.25, 0)
    logit += rng.normal(0, 0.55, n_samples)

    probability = 1 / (1 + np.exp(-logit))
    df["churn"] = (rng.random(n_samples) < probability).astype(int)
    return df


def train_models():
    global rf_model, gb_model, feature_columns, analytics_cache, training_summary

    df = generate_synthetic_dataset()
    feature_columns = ALL_FEATURE_COLUMNS
    X = df[feature_columns]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=260,
                    max_depth=12,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_model.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    if xgb is not None:
        gb_estimator = xgb.XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    else:
        gb_estimator = GradientBoostingClassifier(random_state=42)

    gb_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", gb_estimator),
        ]
    )
    gb_model.fit(X_train, y_train)
    gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])

    analytics_cache = {
        "total_customers": int(len(df)),
        "churn_rate": round(float(df["churn"].mean() * 100), 1),
        "avg_monthly_revenue": int(round(df["monthlyCharges"].mean())),
        "customer_lifetime": int(round(df["tenure"].mean())),
        "churn_by_service": {
            key: round(float(val * 100), 1)
            for key, val in df.groupby("internet")["churn"].mean().rename(
                {"fiber": "Fiber Optic", "dsl": "DSL Internet", "no": "No Internet"}
            ).to_dict().items()
        },
        "churn_by_contract": {
            "Month-to-month": round(float(df.loc[df["contract"] == "monthly", "churn"].mean() * 100), 1),
            "One year": round(float(df.loc[df["contract"] == "yearly", "churn"].mean() * 100), 1),
            "Two year": round(float(df.loc[df["contract"] == "twoyear", "churn"].mean() * 100), 1),
        },
        "insights": [
            "Competitor interaction, contract type, and payment issues dominate early churn risk.",
            "Decreasing usage plus low satisfaction is a stronger churn signal than price alone.",
            "Auto-pay and support add-ons consistently reduce churn probability in the synthetic cohort.",
            "Long-tenure accounts with loyalty membership show materially lower churn even at higher ARPU.",
        ],
    }

    training_summary = {
        "random_forest_auc": round(float(rf_auc), 4),
        "boosted_model_auc": round(float(gb_auc), 4),
    }
    print("Models trained successfully:", training_summary)


def build_reason_list(customer):
    reasons = []

    def add(score, title, description):
        reasons.append((score, title, description))

    if customer["contract"] == "monthly":
        add(0.95, "Month-to-month contract", "Short-term contracts show the highest churn sensitivity.")
    if customer["competitorInteraction"] == "yes":
        add(0.92, "Competitor engagement", "Recent competitive exposure sharply increases switching risk.")
    if customer["planDowngrade"] == "yes":
        add(0.86, "Plan downgrade detected", "Downgrades often precede cancellation or dormancy.")
    if customer["paymentHistory"] == "gt_2_missed":
        add(0.9, "Repeated missed payments", "Frequent billing misses are a strong involuntary churn signal.")
    elif customer["paymentHistory"] == "1_2_missed":
        add(0.62, "Recent payment misses", "Payment friction is already visible on the account.")
    if customer["serviceCalls"] >= 4:
        add(0.78, "High service-call volume", "Repeated support contacts usually indicate unresolved friction.")
    if customer["numComplaints"] >= 2:
        add(0.75, "Complaint count elevated", "Multiple complaints materially raise churn probability.")
    if customer["daysSinceActivity"] > 21 or customer["inactivityPeriod"] > 14:
        add(0.84, "Usage inactivity", "Sustained inactivity is one of the clearest pre-churn behaviors.")
    if customer["usageTrend"] == "decreasing":
        add(0.68, "Usage is declining", "A falling engagement trend often comes before churn.")
    if customer["satisfactionScore"] <= 2:
        add(0.88, "Low satisfaction", "Low CSAT is strongly aligned with churn outcomes.")
    if customer["nps"] <= 4:
        add(0.7, "Low promoter score", "Detractors have materially higher retention risk.")
    if customer["outstandingBalance"] > 1500:
        add(0.64, "Outstanding balance", "Pending balance increases the risk of involuntary churn.")
    if customer["internet"] == "fiber" and customer["techSupport"] == "no":
        add(0.58, "Fiber without support", "High-speed users without support are more vulnerable to frustration.")
    if customer["tenure"] < 12:
        add(0.57, "Early lifecycle account", "Newer customers have not fully formed a retention habit yet.")
    if customer["offerResponse"] == "unsubscribed":
        add(0.56, "Marketing disengagement", "Unsubscribing from offers usually signals declining brand interest.")

    return [
        {"title": title, "description": description}
        for _, title, description in sorted(reasons, key=lambda item: item[0], reverse=True)[:3]
    ]


def get_churn_suggestions(customer, probability):
    suggestions = []

    if probability >= 0.75:
        suggestions.append(
            {
                "priority": "Critical",
                "title": "Immediate retention outreach",
                "description": "Trigger a live retention call within 24 hours with a save offer and issue review.",
                "icon": "solar:shield-warning-linear",
                "colorClass": "text-red-500",
            }
        )
    elif probability >= 0.45:
        suggestions.append(
            {
                "priority": "Medium",
                "title": "Proactive check-in",
                "description": "Schedule a satisfaction touchpoint and present a personalized value reminder.",
                "icon": "solar:chat-round-line-linear",
                "colorClass": "text-amber-500",
            }
        )

    if customer["contract"] == "monthly":
        suggestions.append(
            {
                "priority": "High",
                "title": "Move to longer contract",
                "description": "Offer a 1-year plan discount or bonus bundle to reduce switching intent.",
                "icon": "solar:calendar-minimalistic-linear",
                "colorClass": "text-red-500",
            }
        )

    if customer["payment"] == "electronic" or customer["autoPay"] == "no":
        suggestions.append(
            {
                "priority": "Medium",
                "title": "Reduce payment friction",
                "description": "Promote auto-pay with a one-time credit and simplify payment recovery flows.",
                "icon": "solar:wallet-linear",
                "colorClass": "text-zinc-700",
            }
        )

    if customer["internet"] == "fiber" and customer["techSupport"] == "no":
        suggestions.append(
            {
                "priority": "Medium",
                "title": "Bundle technical support",
                "description": "Add support or premium care for Fiber customers showing high service dependency.",
                "icon": "solar:settings-minimalistic-linear",
                "colorClass": "text-zinc-700",
            }
        )

    if customer["usageTrend"] == "decreasing" or customer["daysSinceActivity"] > 14:
        suggestions.append(
            {
                "priority": "Medium",
                "title": "Re-engagement campaign",
                "description": "Use targeted benefits or plan optimization to reverse declining account activity.",
                "icon": "solar:chart-square-linear",
                "colorClass": "text-zinc-700",
            }
        )

    if customer["satisfactionScore"] <= 2 or customer["numComplaints"] >= 2:
        suggestions.append(
            {
                "priority": "High",
                "title": "Resolve experience issues",
                "description": "Assign a specialist to address complaints and close unresolved support gaps.",
                "icon": "solar:headphones-round-linear",
                "colorClass": "text-amber-500",
            }
        )

    if customer["loyaltyMember"] == "no" and customer["tenure"] >= 12:
        suggestions.append(
            {
                "priority": "Low",
                "title": "Enroll in loyalty plan",
                "description": "Reward stable customers with loyalty benefits before they become price sensitive.",
                "icon": "solar:star-linear",
                "colorClass": "text-zinc-700",
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "priority": "Healthy",
                "title": "Maintain current engagement",
                "description": "This profile looks stable. Keep the account on the standard retention journey.",
                "icon": "solar:shield-check-linear",
                "colorClass": "text-emerald-500",
            }
        )

    return suggestions[:3]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json(force=True) or {}
        customer = build_customer_payload(request_data)
        input_df = pd.DataFrame([[customer[col] for col in feature_columns]], columns=feature_columns)

        rf_probability = float(rf_model.predict_proba(input_df)[0][1])
        gb_probability = float(gb_model.predict_proba(input_df)[0][1])
        avg_probability = (rf_probability + gb_probability) / 2
        percentage = round(avg_probability * 100, 1)

        if percentage >= 70:
            risk_level = "Critical Risk"
            risk_class = "critical"
        elif percentage >= 40:
            risk_level = "Elevated Risk"
            risk_class = "elevated"
        else:
            risk_level = "Healthy Account"
            risk_class = "healthy"

        return jsonify(
            {
                "success": True,
                "risk_percentage": percentage,
                "risk_level": risk_level,
                "risk_class": risk_class,
                "rf_probability": round(rf_probability * 100, 1),
                "xgb_probability": round(gb_probability * 100, 1),
                "top_reasons": build_reason_list(customer),
                "suggestions": get_churn_suggestions(customer, avg_probability),
                "training_summary": training_summary,
            }
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/analytics")
def analytics():
    return jsonify(analytics_cache)


@app.route("/recommendations")
def recommendations():
    return jsonify(
        {
            "strategies": [
                {
                    "title": "Contract Optimization",
                    "impact": "High",
                    "description": "Monthly users are the largest synthetic churn pocket. Move them to annual plans.",
                    "actions": ["Offer 10-15% off on 1-year terms", "Bundle premium support into renewals"],
                },
                {
                    "title": "Payment Reliability",
                    "impact": "High",
                    "description": "Missed-payment accounts churn much faster than auto-pay accounts.",
                    "actions": ["Promote bank/card auto-pay", "Prioritize outstanding-balance recovery"],
                },
                {
                    "title": "Experience Recovery",
                    "impact": "Medium",
                    "description": "Complaint-heavy and low-CSAT users need fast operational recovery.",
                    "actions": ["Escalate repeat complaints", "Offer issue-resolution follow-up calls"],
                },
            ]
        }
    )


if __name__ == "__main__":
    train_models()
    app.run(debug=True, host="0.0.0.0", port=5050)
else:
    train_models()
