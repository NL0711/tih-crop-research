"""
compare_experiments.py
======================
CLI tool to compare multiple DAMamba experiments.

Usage
-----
    python compare_experiments.py
    python compare_experiments.py --summary-csv experiments/experiment_summary.csv
    python compare_experiments.py --summary-csv experiments/experiment_summary.csv --sort best_val_accuracy
"""

from __future__ import annotations

import argparse, csv, json, logging, os, sys
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


DISPLAY_COLS = [
    "experiment_name", "timestamp", "architecture", "hybrid_enabled",
    "best_val_accuracy", "best_epoch", "final_val_accuracy",
    "test_accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall", "kappa", "mcc",
    "macro_auc", "ece", "parameter_count",
    "training_time_s", "mean_inference_ms",
]

NUMERIC_COLS = {
    "best_val_accuracy", "final_val_accuracy", "test_accuracy",
    "balanced_accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall", "kappa", "mcc",
    "macro_auc", "ece", "training_time_s", "mean_inference_ms",
    "parameter_count", "best_epoch",
}


def load_summary(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        log.error(f"Summary file not found: {path}")
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in NUMERIC_COLS:
                if k in row:
                    try:
                        row[k] = float(row[k]) if row[k] != "" else None
                    except ValueError:
                        row[k] = None
            rows.append(row)
    return rows


def print_table(rows: List[Dict[str, Any]], cols: List[str]) -> None:
    if not rows:
        print("No experiments found.")
        return
    # Filter to existing columns
    avail = [c for c in cols if any(c in r for r in rows)]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in avail}
    sep = "+-" + "-+-".join("-" * widths[c] for c in avail) + "-+"
    hdr = "| " + " | ".join(c.ljust(widths[c]) for c in avail) + " |"
    print(sep); print(hdr); print(sep)
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in avail) + " |"
        print(line)
    print(sep)


def save_comparison_plot(rows: List[Dict[str, Any]], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [r.get("experiment_name", f"exp{i}") for i, r in enumerate(rows)]
        vals   = [r.get("best_val_accuracy") for r in rows]
        test   = [r.get("test_accuracy")     for r in rows]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
        bars1 = ax.bar(x - width/2,
                       [v if v is not None else 0 for v in vals],
                       width, label="Best Val Acc@1 (%)")
        bars2 = ax.bar(x + width/2,
                       [v if v is not None else 0 for v in test],
                       width, label="Test Acc@1 (%)", color="orange")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)"); ax.set_title("Experiment Comparison")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        log.info(f"Comparison plot saved: {out_path}")
    except Exception as exc:
        log.warning(f"Could not create comparison plot: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DAMamba experiments")
    parser.add_argument(
        "--summary-csv",
        default="experiments/experiment_summary.csv",
        help="Path to experiment_summary.csv (default: experiments/experiment_summary.csv)",
    )
    parser.add_argument(
        "--sort",
        default="best_val_accuracy",
        help="Column to sort by (descending). Default: best_val_accuracy",
    )
    parser.add_argument(
        "--out-plot",
        default=None,
        help="Optional path to save a comparison bar chart PNG",
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        default=None,
        help="Subset of columns to display",
    )
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    if not rows:
        sys.exit(1)

    # Sort
    sort_col = args.sort
    rows.sort(key=lambda r: (r.get(sort_col) is None, -(r.get(sort_col) or 0)))

    cols = args.cols if args.cols else DISPLAY_COLS
    print(f"\n=== Experiment Comparison ({len(rows)} experiments, sorted by {sort_col}) ===\n")
    print_table(rows, cols)

    # Plot
    out_plot = args.out_plot or os.path.join(
        os.path.dirname(args.summary_csv), "comparison_plot.png"
    )
    save_comparison_plot(rows, out_plot)

    # Best experiment summary
    best = rows[0]
    print(f"\nBest experiment : {best.get('experiment_name', 'N/A')}")
    print(f"  Timestamp     : {best.get('timestamp', 'N/A')}")
    print(f"  Val Acc@1     : {best.get('best_val_accuracy', 'N/A')}")
    print(f"  Test Acc@1    : {best.get('test_accuracy', 'N/A')}")
    print(f"  Macro F1      : {best.get('macro_f1', 'N/A')}")
    print(f"  Exp dir       : {best.get('exp_dir', 'N/A')}")


if __name__ == "__main__":
    main()
