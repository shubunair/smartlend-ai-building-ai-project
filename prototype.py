"""
Educational prototype for SmartLend AI.

IMPORTANT:
- Uses synthetic data only.
- Not validated for lending.
- Must not be used to approve, decline, price, or otherwise make real credit decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def make_synthetic_data(n_rows: int = 2500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic small-business lending dataset."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "annual_revenue": rng.lognormal(mean=np.log(550_000), sigma=0.75, size=n_rows),
            "avg_monthly_balance": rng.lognormal(mean=np.log(35_000), sigma=0.9, size=n_rows),
            "debt_service_monthly": rng.lognormal(mean=np.log(8_000), sigma=0.65, size=n_rows),
            "credit_score": np.clip(rng.normal(690, 65, n_rows), 500, 850),
            "months_in_business": np.clip(rng.gamma(4.5, 14, n_rows), 3, 360),
            "nsf_count_90d": rng.poisson(0.8, n_rows),
            "prior_default": rng.binomial(1, 0.08, n_rows),
            "requested_amount": rng.lognormal(mean=np.log(120_000), sigma=0.7, size=n_rows),
        }
    )

    revenue_monthly = df["annual_revenue"] / 12
    debt_burden = df["debt_service_monthly"] / np.maximum(revenue_monthly, 1)
    request_ratio = df["requested_amount"] / np.maximum(df["annual_revenue"], 1)

    logit = (
        -2.4
        + 2.8 * debt_burden
        + 1.6 * request_ratio
        - 0.007 * (df["credit_score"] - 650)
        - 0.006 * (df["months_in_business"] - 24)
        - 0.000012 * df["avg_monthly_balance"]
        + 0.28 * df["nsf_count_90d"]
        + 1.4 * df["prior_default"]
    )

    probability = 1 / (1 + np.exp(-np.clip(logit, -12, 12)))
    df["serious_delinquency"] = rng.binomial(1, probability)
    return df


def build_model(features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, features)])
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    data = make_synthetic_data()

    features = [
        "annual_revenue",
        "avg_monthly_balance",
        "debt_service_monthly",
        "credit_score",
        "months_in_business",
        "nsf_count_90d",
        "prior_default",
        "requested_amount",
    ]

    X = data[features]
    y = data["serious_delinquency"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_model(features)
    pipeline.fit(X_train, y_train)

    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    print("SmartLend AI - Educational Prototype")
    print("=" * 42)
    print(f"Rows: {len(data):,}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, probability):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, prediction, digits=3))

    example = pd.DataFrame([
        {
            "annual_revenue": 720_000,
            "avg_monthly_balance": 48_000,
            "debt_service_monthly": 9_500,
            "credit_score": 705,
            "months_in_business": 72,
            "nsf_count_90d": 0,
            "prior_default": 0,
            "requested_amount": 125_000,
        }
    ])

    example_risk = pipeline.predict_proba(example)[:, 1][0]

    print("\nExample synthetic application")
    print(example.to_string(index=False))
    print(f"\nEstimated synthetic delinquency probability: {example_risk:.1%}")
    print(
        "\nThis score is for demonstration only. "
        "A real lending system requires validated data, fairness testing, "
        "regulatory review, model governance, and human accountability."
    )


if __name__ == "__main__":
    main()
