"""Charts for the credit model: discrimination, separation, and calibration.

Matplotlib only. Each function returns the Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from . import metrics


def plot_roc(y_true, pd_hat, title="ROC curve"):
    fpr, tpr, _ = roc_curve(y_true, pd_hat)
    a = metrics.auc(y_true, pd_hat)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"Model (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ks(y_true, pd_hat, title="KS separation"):
    """Cumulative payer and defaulter curves, with the KS gap marked."""
    y_true = np.asarray(y_true)
    pd_hat = np.asarray(pd_hat)
    order = np.argsort(pd_hat)
    y_sorted = y_true[order]
    pd_sorted = pd_hat[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    cum_bad = np.cumsum(y_sorted) / n_bad
    cum_good = np.cumsum(1 - y_sorted) / n_good
    gap = np.abs(cum_good - cum_bad)
    k = int(np.argmax(gap))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pd_sorted, cum_good, label="Payers", color="tab:green")
    ax.plot(pd_sorted, cum_bad, label="Defaulters", color="tab:red")
    ax.vlines(pd_sorted[k], cum_bad[k], cum_good[k], color="black",
              linestyle=":", label=f"KS = {gap[k]:.3f}")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Cumulative share")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores, y_true, title="Credit score distribution"):
    """Overlaid score histograms for payers and defaulters."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="Paid", color="tab:green")
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.6, label="Default", color="tab:red")
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Applicants")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calibration(y_true, pd_hat, n_bins=10, title="Calibration"):
    table = metrics.calibration_table(y_true, pd_hat, n_bins)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(table["predicted_default_rate"], table["actual_default_rate"],
            "o-", color="tab:blue", label="Model")
    lim = [0, max(table["predicted_default_rate"].max(), table["actual_default_rate"].max()) * 1.05]
    ax.plot(lim, lim, "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted default rate")
    ax.set_ylabel("Actual default rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
