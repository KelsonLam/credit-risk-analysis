"""Data for the credit model: synthetic by default, your own CSV if you have one.

The synthetic generator is not random noise. Each applicant's features are drawn
from plausible distributions, and the probability of default is then built from
those features through a logistic link with sensible signs: more delinquencies,
higher debt-to-income, higher utilization, lower income, and shorter job tenure
all raise default risk. That means a model fit to this data should recover the
relationships, which is what makes it a fair test of the pipeline.

The data is clearly synthetic and is meant for demonstrating the modelling, not
for drawing conclusions about real borrowers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "age",
    "annual_income",
    "debt_to_income",
    "credit_utilization",
    "num_delinquencies",
    "employment_length",
    "loan_amount",
    "num_open_accounts",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(linear: np.ndarray, target_rate: float) -> float:
    """Find the intercept so the average default probability hits the target."""
    lo, hi = -15.0, 15.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _sigmoid(linear + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_synthetic(
    n_samples: int = 12000, default_rate: float = 0.18, seed: int = 42
) -> pd.DataFrame:
    """Build a synthetic loan book with a feature-driven default flag."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n_samples).astype(float)
    annual_income = np.clip(rng.lognormal(10.9, 0.5, n_samples), 12_000, 500_000)
    debt_to_income = np.clip(rng.beta(2.0, 5.0, n_samples), 0.0, 0.95)
    credit_utilization = np.clip(rng.beta(2.0, 3.0, n_samples), 0.0, 1.0)
    num_delinquencies = rng.poisson(0.6, n_samples).astype(float)
    employment_length = np.clip(rng.exponential(6.0, n_samples), 0.0, 40.0)
    loan_amount = np.clip(rng.lognormal(9.4, 0.6, n_samples), 1_000, 100_000)
    num_open_accounts = np.clip(rng.poisson(8, n_samples), 0, 40).astype(float)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "credit_utilization": credit_utilization,
        "num_delinquencies": num_delinquencies,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "num_open_accounts": num_open_accounts,
    })

    # Standardize for a stable linear predictor, then apply signed weights.
    z = (df - df.mean()) / df.std(ddof=0)
    linear = (
        1.15 * z["debt_to_income"]
        + 1.05 * z["credit_utilization"]
        + 0.95 * z["num_delinquencies"]
        - 0.70 * np.log(df["annual_income"]).pipe(lambda s: (s - s.mean()) / s.std(ddof=0))
        - 0.45 * z["employment_length"]
        + 0.40 * z["loan_amount"]
        - 0.25 * z["age"]
    ).to_numpy()

    intercept = _solve_intercept(linear, default_rate)
    prob_default = _sigmoid(linear + intercept)
    df["default"] = rng.binomial(1, prob_default)
    return df


def load_data(
    csv_path: str | Path | None,
    target_column: str = "default",
    n_samples: int = 12000,
    default_rate: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a CSV if one is given and exists, otherwise generate synthetic data."""
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            df = pd.read_csv(path)
            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found in {path}."
                )
            return df
    return generate_synthetic(n_samples, default_rate, seed)
