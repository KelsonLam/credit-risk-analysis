"""Command line entry point: train the credit model and report on it.

Examples
--------
Run on the built-in synthetic data::

    python scripts/run_credit_model.py

Point it at your own loan CSV (target column named in config.yaml)::

    python scripts/run_credit_model.py --csv data/loans.csv

Save the charts::

    python scripts/run_credit_model.py --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk import data, model, metrics, scorecard, plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate a credit risk model.")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    p.add_argument("--csv", help="Path to a loan CSV (overrides config).")
    p.add_argument("--save-plots", action="store_true", help="Write charts to results/.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dcfg, mcfg, scfg = cfg["data"], cfg["model"], cfg["scorecard"]

    csv_path = args.csv if args.csv is not None else dcfg.get("csv_path") or None
    df = data.load_data(
        csv_path, dcfg["target_column"],
        dcfg["n_samples"], dcfg["default_rate"], dcfg["seed"],
    )
    source = csv_path if csv_path else "synthetic data"
    print(f"Loaded {len(df):,} rows from {source}.")

    res = model.train_model(df, dcfg["target_column"], mcfg["test_size"], dcfg["seed"])

    stats = metrics.summarize(res.y_test, res.pd_test, mcfg["threshold"])
    print("\nModel performance (test set)")
    print("-" * 32)
    for k, v in stats.items():
        print(f"{k:<14} {v:,.4f}")

    print("\nClassification report at threshold "
          f"{mcfg['threshold']}:")
    print(metrics.report_at(res.y_test, res.pd_test, mcfg["threshold"]))

    table = scorecard.scorecard_table(
        res.pd_test, scfg["base_score"], scfg["base_odds"], scfg["points_to_double_odds"]
    )
    table["actual_default"] = res.y_test.to_numpy()
    print("Risk bands (test set)")
    print("-" * 32)
    print(scorecard.band_summary(table).to_string())

    if args.save_plots:
        figs = {
            "roc": plotting.plot_roc(res.y_test, res.pd_test),
            "ks": plotting.plot_ks(res.y_test, res.pd_test),
            "score_distribution": plotting.plot_score_distribution(table["score"], res.y_test),
            "calibration": plotting.plot_calibration(res.y_test, res.pd_test),
        }
        for name, fig in figs.items():
            out = plotting.save_figure(fig, f"results/{name}.png")
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
