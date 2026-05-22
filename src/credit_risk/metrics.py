"""How good is the model at separating defaulters from everyone else?

Credit modelling leans on a specific set of measures:

    AUC   probability a random defaulter is ranked riskier than a random payer
    Gini  2 * AUC - 1, the same information on the 0 to 1 scale lenders quote
    KS    the largest gap between the score distributions of payers and
          defaulters, the classic "how separable are the two groups" number
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def auc(y_true, pd_hat) -> float:
    return float(roc_auc_score(y_true, pd_hat))


def gini(y_true, pd_hat) -> float:
    return 2.0 * auc(y_true, pd_hat) - 1.0


def ks_statistic(y_true, pd_hat) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative payer and defaulter rates."""
    y_true = np.asarray(y_true)
    pd_hat = np.asarray(pd_hat)
    order = np.argsort(pd_hat)
    y_sorted = y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    if n_bad == 0 or n_good == 0:
        return float("nan")
    cum_bad = np.cumsum(y_sorted) / n_bad
    cum_good = np.cumsum(1 - y_sorted) / n_good
    return float(np.max(np.abs(cum_good - cum_bad)))


def confusion_at(y_true, pd_hat, threshold: float = 0.5) -> np.ndarray:
    preds = (np.asarray(pd_hat) >= threshold).astype(int)
    return confusion_matrix(y_true, preds)


def report_at(y_true, pd_hat, threshold: float = 0.5) -> str:
    preds = (np.asarray(pd_hat) >= threshold).astype(int)
    return classification_report(y_true, preds, target_names=["Paid", "Default"])


def calibration_table(y_true, pd_hat, n_bins: int = 10) -> pd.DataFrame:
    """Bucket by predicted PD and compare predicted vs actual default rate.

    A well-calibrated model has predicted and actual close in every bucket.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    df["bucket"] = pd.qcut(df["pd"], q=n_bins, duplicates="drop")
    table = df.groupby("bucket", observed=True).agg(
        count=("y", "size"),
        predicted_default_rate=("pd", "mean"),
        actual_default_rate=("y", "mean"),
    )
    return table.reset_index()


def summarize(y_true, pd_hat, threshold: float = 0.5) -> dict[str, float]:
    return {
        "AUC": auc(y_true, pd_hat),
        "Gini": gini(y_true, pd_hat),
        "KS statistic": ks_statistic(y_true, pd_hat),
        "Default rate": float(np.mean(y_true)),
    }
