#!/usr/bin/env python3
"""
========================================================================================
Plot Model Performance Comparison (Grouped Bar Chart)
========================================================================================
Generates publication-quality comparison charts for:
  - ResNet-50
  - EfficientNet-B1
  - Inception-ResNet-v2
  - Inception-v3
  - VMamba-T
  - DAMamba-Tiny
  - DAMamba-CNN (Proposed Hybrid)

Evaluated across:
  - Accuracy (in %)
  - Precision (in %)
  - Recall (in %)
  - Specificity (in %)
  - F1-Score (in %)

Usage:
  python plot_model_comparison.py
  python plot_model_comparison.py --theme paper_orange
  python plot_model_comparison.py --theme academic
  python plot_model_comparison.py --theme highlight
========================================================================================
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Default model display names and fallback data
# -----------------------------------------------------------------------------
DEFAULT_METRICS_DATA = {
    "ResNet-50": {
        "dir_name": "resnet50",
        "Accuracy": 86.01,
        "Precision": 88.98,
        "Recall": 88.64,
        "Specificity": 96.38,
        "F1-Score": 88.06,
    },
    "EfficientNet-B1": {
        "dir_name": "EfficientNetB1",
        "Accuracy": 86.01,
        "Precision": 87.07,
        "Recall": 86.92,
        "Specificity": 96.44,
        "F1-Score": 86.73,
    },
    "Inception-v3": {
        "dir_name": "inception_v3",
        "Accuracy": 84.62,
        "Precision": 87.19,
        "Recall": 85.39,
        "Specificity": 96.10,
        "F1-Score": 85.25,
    },
    "Inception-ResNet-v2": {
        "dir_name": "InceptioResNetV2",
        "Accuracy": 88.81,
        "Precision": 91.23,
        "Recall": 90.31,
        "Specificity": 97.15,
        "F1-Score": 90.05,
    },
    "VMamba-T": {
        "dir_name": "VMamba-T",
        "Accuracy": 71.33,
        "Precision": 79.29,
        "Recall": 76.36,
        "Specificity": 92.83,
        "F1-Score": 75.46,
    },
    "DAMamba-Tiny": {
        "dir_name": "damamba-tiny",
        "Accuracy": 96.50,
        "Precision": 96.97,
        "Recall": 97.47,
        "Specificity": 99.10,
        "F1-Score": 97.11,
    },
    "DAMamba-CNN (Proposed)": {
        "dir_name": "damamba-cnn",
        "Accuracy": 97.78,
        "Precision": 97.85,
        "Recall": 97.60,
        "Specificity": 99.35,
        "F1-Score": 97.70,
    },
}

METRIC_NAMES = [
    "Accuracy (in %)",
    "Precision (in %)",
    "Recall (in %)",
    "Specificity (in %)",
    "F1-Score (in %)",
]

METRIC_KEYS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]


def extract_metrics_from_forgraph(forgraph_dir: str):
    """
    Scans the ForGraph directory and extracts computed metrics where available.
    Falls back to curated dataset values if specific test evaluation files are absent.
    """
    metrics_data = {}
    for display_name, defaults in DEFAULT_METRICS_DATA.items():
        dir_name = defaults.get("dir_name", "")
        model_path = os.path.join(forgraph_dir, dir_name)
        
        entry = {
            "Accuracy": defaults["Accuracy"],
            "Precision": defaults["Precision"],
            "Recall": defaults["Recall"],
            "Specificity": defaults["Specificity"],
            "F1-Score": defaults["F1-Score"],
        }
        
        # Check test_eval JSON
        test_m_path = os.path.join(model_path, "test_eval", "test_metrics.json")
        per_class_path = os.path.join(model_path, "test_eval", "metrics", "per_class_metrics.csv")
        
        if os.path.exists(test_m_path):
            try:
                with open(test_m_path, "r") as f:
                    tm = json.load(f)
                if "top1_accuracy" in tm:
                    entry["Accuracy"] = round(float(tm["top1_accuracy"]), 2)
                if "macro_precision" in tm:
                    entry["Precision"] = round(float(tm["macro_precision"]) * 100, 2)
                if "macro_recall" in tm:
                    entry["Recall"] = round(float(tm["macro_recall"]) * 100, 2)
                if "macro_f1" in tm:
                    entry["F1-Score"] = round(float(tm["macro_f1"]) * 100, 2)
            except Exception as e:
                print(f"Warning reading {test_m_path}: {e}")
                
        if os.path.exists(per_class_path):
            try:
                with open(per_class_path, "r") as f:
                    reader = csv.DictReader(f)
                    specs = [float(row["specificity"]) for row in reader if row.get("specificity")]
                    if specs:
                        entry["Specificity"] = round((sum(specs) / len(specs)) * 100, 2)
            except Exception as e:
                print(f"Warning reading {per_class_path}: {e}")
                
        metrics_data[display_name] = entry

    return metrics_data


def get_color_palette(theme: str, num_models: int):
    """
    Returns harmonious color palettes tailored for publications.
    """
    if theme == "paper_orange":
        # Gradation of warm terracotta/orange tones as in the reference image
        # From dark amber/brown to soft peach/rose
        palette = [
            "#A0522D",  # Sienna / Dark Bronze (ResNet-50)
            "#CD6600",  # Dark Orange (EfficientNet-B1)
            "#D97724",  # Ochre / Amber (Inception-v3)
            "#E68A4E",  # Warm Terracotta (Inception-ResNet-v2)
            "#C98A75",  # Muted Rose / VMamba
            "#F0A882",  # Salmon Peach (DAMamba-Tiny)
            "#F8BFA4",  # Light Apricot / Peach (Proposed DAMamba-CNN)
        ]
    elif theme == "academic":
        # Distinct high-visibility journal color scheme
        palette = [
            "#2B5C8F",  # Classic Navy (ResNet-50)
            "#4E79A7",  # Steel Blue (EfficientNet-B1)
            "#59A14F",  # Sage Green (Inception-v3)
            "#F28E2B",  # Orange (Inception-ResNet-v2)
            "#807DBA",  # Slate Purple (VMamba-T)
            "#E15759",  # Coral Red (DAMamba-Tiny)
            "#B07AA1",  # Violet (DAMamba-CNN)
        ]
    elif theme == "highlight":
        # Cool grey/slate tones for baselines, vibrant teal & coral for DAMamba
        palette = [
            "#7F8C8D",  # Gray (ResNet-50)
            "#95A5A6",  # Light Gray (EfficientNet-B1)
            "#5D6D7E",  # Slate (Inception-v3)
            "#34495E",  # Charcoal (Inception-ResNet-v2)
            "#85929E",  # Cool Steel (VMamba-T)
            "#E67E22",  # Radiant Orange (DAMamba-Tiny)
            "#E74C3C",  # Vibrant Red (DAMamba-CNN)
        ]
    else:  # 'nature'
        palette = [
            "#3B528B",  # Indigo
            "#21908C",  # Teal
            "#5DC863",  # Light Green
            "#E78429",  # Warm Amber
            "#88419D",  # Violet
            "#CF597E",  # Rose Red
            "#EE8363",  # Peach Red
        ]

    if num_models <= len(palette):
        return palette[:num_models]
    return plt.cm.tab10(np.linspace(0, 1, num_models))


def plot_model_comparison(
    metrics_data: dict,
    output_path: str,
    theme: str = "paper_orange",
    fig_title: str = "Figure 12. The performance of models evaluated across different metrics",
    y_min: float = 65.0,
    y_max: float = 101.0,
    show_values: bool = False,
    include_caption: bool = True,
):
    """
    Renders and saves the grouped bar comparison plot.
    """
    # Publication font settings
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Times New Roman"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.9

    models = list(metrics_data.keys())
    num_models = len(models)
    num_metrics = len(METRIC_NAMES)
    
    # Coordinates
    x = np.arange(num_metrics)  # 0, 1, 2, 3, 4
    
    # Calculate bar width and offsets
    total_group_width = 0.76
    bar_width = total_group_width / num_models
    
    # Colors
    colors = get_color_palette(theme, num_models)
    
    # Create Figure
    fig_width = 10.5 if num_models >= 6 else 8.5
    fig_height = 6.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    
    # Plot each model's bars
    for idx, model_name in enumerate(models):
        values = [metrics_data[model_name][k] for k in METRIC_KEYS]
        offset = (idx - (num_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            label=model_name,
            color=colors[idx],
            edgecolor="#444444",
            linewidth=0.5,
            zorder=3,
        )
        
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

    # Grid & Limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.55, num_metrics - 0.45)
    ax.set_ylabel("Values", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlabel("Evaluation Metric", fontsize=13, fontweight="bold", labelpad=12)
    
    # Ticks formatting
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=11, fontweight="medium", rotation=40, ha="right")
    ax.tick_params(axis="y", labelsize=11, direction="out", length=5, width=0.8)
    ax.tick_params(axis="x", direction="out", length=5, width=0.8)

    # Subtle horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#888888", zorder=0)
    ax.set_axisbelow(True)

    # Legend on the right side
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=10.5,
        title_fontsize=11,
        handlelength=1.3,
        handleheight=1.0,
        labelspacing=0.7,
    )

    # Tight layout with room for right legend and bottom caption
    plt.tight_layout(rect=[0, 0.08 if include_caption else 0, 1, 1])

    # Add Figure Caption underneath the chart
    if include_caption and fig_title:
        fig.text(
            0.5,
            0.02,
            fig_title,
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontfamily="sans-serif",
        )

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Also save as PDF for publication
    pdf_path = str(Path(output_path).with_suffix(".pdf"))
    fig.savefig(pdf_path, bbox_inches="tight")
    
    plt.close(fig)
    print(f"[Success] Saved figure to: {output_path}")
    print(f"[Success] Saved PDF (vector format) to: {pdf_path}")


def save_metrics_summary_table(metrics_data: dict, out_csv_path: str):
    """
    Saves a tabular summary CSV and prints a clean formatted table in console.
    """
    headers = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "Specificity (%)", "F1-Score (%)"]
    rows = []
    
    print("\n" + "=" * 86)
    print(f"{'MODEL PERFORMANCE EVALUATION SUMMARY':^86}")
    print("=" * 86)
    print(f" {'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<11} | {'F1-Score':<10} ")
    print("-" * 86)
    
    for model, m in metrics_data.items():
        row = [
            model,
            f"{m['Accuracy']:.2f}",
            f"{m['Precision']:.2f}",
            f"{m['Recall']:.2f}",
            f"{m['Specificity']:.2f}",
            f"{m['F1-Score']:.2f}",
        ]
        rows.append(row)
        print(f" {model:<25} | {m['Accuracy']:>9.2f}% | {m['Precision']:>9.2f}% | {m['Recall']:>9.2f}% | {m['Specificity']:>10.2f}% | {m['F1-Score']:>9.2f}% ")
        
    print("=" * 86 + "\n")
    
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"[Success] Saved metrics CSV summary to: {out_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Model Comparison Grouped Bar Chart")
    parser.add_argument(
        "--forgraph-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ForGraph"),
        help="Path to ForGraph folder containing model subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ForGraph", "model_performance_comparison.png"),
        help="Path to save output image (.png)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="paper_orange",
        choices=["paper_orange", "academic", "highlight", "nature"],
        help="Color theme for the plot bars",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Figure 12. The performance of models evaluated across different metrics",
        help="Bottom caption text for the figure",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=68.0,
        help="Y-axis minimum value",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=102.0,
        help="Y-axis maximum value",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Annotate exact numeric values on top of each bar",
    )
    args = parser.parse_args()

    # 1. Extract or compile metrics
    if os.path.isdir(args.forgraph_dir):
        print(f"Reading metrics from: {args.forgraph_dir}")
        metrics_data = extract_metrics_from_forgraph(args.forgraph_dir)
    else:
        print(f"ForGraph directory not found at {args.forgraph_dir}. Using default project metrics.")
        metrics_data = {k: {m: DEFAULT_METRICS_DATA[k][m] for m in METRIC_KEYS} for k in DEFAULT_METRICS_DATA}

    # 2. Save CSV summary
    csv_out = str(Path(args.output).with_name("model_metrics_summary.csv"))
    save_metrics_summary_table(metrics_data, csv_out)

    # 3. Generate primary plot
    plot_model_comparison(
        metrics_data=metrics_data,
        output_path=args.output,
        theme=args.theme,
        fig_title=args.caption,
        y_min=args.y_min,
        y_max=args.y_max,
        show_values=args.show_values,
        include_caption=True,
    )

    # 4. Generate alternative themes for convenience
    for alt_theme in ["paper_orange", "academic", "highlight", "nature"]:
        alt_output = str(Path(args.output).with_name(f"model_comparison_{alt_theme}.png"))
        plot_model_comparison(
            metrics_data=metrics_data,
            output_path=alt_output,
            theme=alt_theme,
            fig_title=args.caption,
            y_min=args.y_min,
            y_max=args.y_max,
            show_values=args.show_values,
            include_caption=True,
        )


if __name__ == "__main__":
    main()
