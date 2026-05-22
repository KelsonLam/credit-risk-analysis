"""Preprocessing and the logistic-regression default model.

Logistic regression is the deliberate choice here. A gradient-boosted model
might squeeze out a little more discrimination, but in credit you usually have
to explain why an applicant was declined, and a linear model in log-odds is
about as explainable as it gets. Probabilities are left uncalibrated by the
class weights (no rebalancing), because a scorecard needs the predicted
probability of default to mean what it says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    pipeline: Pipeline
    feature_names: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    pd_test: np.ndarray        # predicted probability of default on the test set
    X_train: pd.DataFrame
    y_train: pd.Series
    pd_train: np.ndarray


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = [("num", StandardScaler(), numeric)]
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def train_model(
    df: pd.DataFrame,
    target_column: str = "default",
    test_size: float = 0.25,
    seed: int = 42,
) -> ModelResult:
    """Split, fit the logistic model, and return the pieces for evaluation."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the data.")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    if y.nunique() < 2:
        raise ValueError("The target must contain both defaults and non-defaults.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = _build_pipeline(X)
    pipe.fit(X_train, y_train)

    return ModelResult(
        pipeline=pipe,
        feature_names=list(X.columns),
        X_test=X_test,
        y_test=y_test,
        pd_test=pipe.predict_proba(X_test)[:, 1],
        X_train=X_train,
        y_train=y_train,
        pd_train=pipe.predict_proba(X_train)[:, 1],
    )
