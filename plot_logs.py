"""
plot_logs.py
============
Parse DAMamba training logs and generate comprehensive, publication-ready plots
for individual experiments and comparative analyses.
"""

import os
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def parse_logs(log_file):
    """
    Parses DAMamba training log files.
    Returns:
        dict containing:
            - train_losses: {epoch: avg_train_loss}
            - train_lrs: {epoch: avg_or_last_lr}
            - train_grad_norms: {epoch: avg_grad_norm}
            - val_losses: {epoch: val_loss}
            - val_ema_losses: {epoch: val_ema_loss}
            - val_acc1: {epoch: val_acc1}
            - val_acc5: {epoch: val_acc5}
            - val_ema_acc1: {epoch: val_ema_acc1}
            - val_ema_acc5: {epoch: val_ema_acc5}
            - epoch_times: {epoch: time_str}
    """
    epochs_data = defaultdict(lambda: {'train_losses': [], 'lrs': [], 'grad_norms': []})
    val_losses = {}
    val_ema_losses = {}
    val_acc1 = {}
    val_acc5 = {}
    val_ema_acc1 = {}
    val_ema_acc5 = {}
    epoch_times = {}

    current_epoch = None
    last_test_loss = None

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Parse train step
            tm = re.search(
                r'Train:\s*\[(\d+)/\d+\]\[\d+/\d+\]\s+eta\s+[^\s]+\s+lr\s+([0-9.e+-]+).*?loss\s+([0-9.]+)(?:\s+\([0-9.]+\))?\s+grad_norm\s+([0-9.e+-]+|nan|inf)',
                line
            )
            if tm:
                ep = int(tm.group(1))
                current_epoch = ep
                lr = float(tm.group(2))
                loss = float(tm.group(3))
                gn_str = tm.group(4)
                gn = float(gn_str) if gn_str not in ('nan', 'inf') else None
                epochs_data[ep]['train_losses'].append(loss)
                epochs_data[ep]['lrs'].append(lr)
                if gn is not None:
                    epochs_data[ep]['grad_norms'].append(gn)

            # Parse epoch time
            etime_m = re.search(r'EPOCH\s+(\d+)\s+training takes\s+(\d+:\d+:\d+)', line)
            if etime_m:
                ep = int(etime_m.group(1))
                epoch_times[ep] = etime_m.group(2)

            # Parse test step loss
            test_m = re.search(r'Test:\s*\[.*?Loss\s+[0-9.]+\s+\(([0-9.]+)\)', line)
            if test_m:
                last_test_loss = float(test_m.group(1))

            # Parse accuracy summary
            acc_m = re.search(r'\*\s+Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)', line)
            if acc_m:
                a1 = float(acc_m.group(1))
                a5 = float(acc_m.group(2))
                if current_epoch is not None:
                    if current_epoch not in val_acc1:
                        val_acc1[current_epoch] = a1
                        val_acc5[current_epoch] = a5
                        if last_test_loss is not None:
                            val_losses[current_epoch] = last_test_loss
                    else:
                        val_ema_acc1[current_epoch] = a1
                        val_ema_acc5[current_epoch] = a5
                        if last_test_loss is not None:
                            val_ema_losses[current_epoch] = last_test_loss

    avg_train_losses = {
        ep: sum(d['train_losses']) / len(d['train_losses'])
        for ep, d in epochs_data.items() if d['train_losses']
    }
    avg_lrs = {
        ep: d['lrs'][-1]
        for ep, d in epochs_data.items() if d['lrs']
    }
    avg_grad_norms = {
        ep: sum(d['grad_norms']) / len(d['grad_norms'])
        for ep, d in epochs_data.items() if d['grad_norms']
    }

    return {
        'train_losses': avg_train_losses,
        'train_lrs': avg_lrs,
        'train_grad_norms': avg_grad_norms,
        'val_losses': val_losses,
        'val_ema_losses': val_ema_losses,
        'val_acc1': val_acc1,
        'val_acc5': val_acc5,
        'val_ema_acc1': val_ema_acc1,
        'val_ema_acc5': val_ema_acc5,
        'epoch_times': epoch_times,
    }


def plot_single_experiment_dashboard(data, title, save_path):
    """
    Generates a 4-panel dashboard for a single experiment:
    1. Training & Validation Loss
    2. Validation Acc@1 & Acc@5 (Standard & EMA)
    3. Best Validation Accuracy Progress
    4. Learning Rate Schedule & Gradient Norm
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # 1. Loss Curves
    ax1 = axs[0, 0]
    ax1.set_title('Loss Curves (Train vs Val)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)

    t_epochs = sorted(data['train_losses'].keys())
    t_losses = [data['train_losses'][e] for e in t_epochs]
    if t_losses:
        ax1.plot(t_epochs, t_losses, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Training Loss')

    v_epochs = sorted(data['val_losses'].keys())
    v_losses = [data['val_losses'][e] for e in v_epochs]
    if v_losses:
        ax1.plot(v_epochs, v_losses, 's-', color='#d62728', linewidth=2, markersize=4, label='Validation Loss')

    ve_epochs = sorted(data['val_ema_losses'].keys())
    ve_losses = [data['val_ema_losses'][e] for e in ve_epochs]
    if ve_losses:
        ax1.plot(ve_epochs, ve_losses, '^--', color='#e377c2', linewidth=1.5, markersize=3, label='EMA Validation Loss')

    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='#f8f9fa')

    # 2. Validation Accuracy Curves
    ax2 = axs[0, 1]
    ax2.set_title('Validation Accuracy Progression', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)

    va_epochs = sorted(data['val_acc1'].keys())
    va_accs = [data['val_acc1'][e] for e in va_epochs]
    if va_accs:
        ax2.plot(va_epochs, va_accs, 'o-', color='#2ca02c', linewidth=2, markersize=4, label='Val Acc@1')
        best_va = max(va_accs)
        best_ep = va_epochs[va_accs.index(best_va)]
        ax2.annotate(f'Best: {best_va:.2f}% (Ep {best_ep})',
                     xy=(best_ep, best_va),
                     xytext=(best_ep, best_va - 4 if best_va > 20 else best_va + 4),
                     arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                     fontweight='bold', fontsize=9, color='#1b5e20')

    vea_epochs = sorted(data['val_ema_acc1'].keys())
    vea_accs = [data['val_ema_acc1'][e] for e in vea_epochs]
    if vea_accs:
        ax2.plot(vea_epochs, vea_accs, '^--', color='#9467bd', linewidth=2, markersize=4, label='EMA Val Acc@1')
        best_vea = max(vea_accs)
        best_vea_ep = vea_epochs[vea_accs.index(best_vea)]
        ax2.annotate(f'EMA Best: {best_vea:.2f}% (Ep {best_vea_ep})',
                     xy=(best_vea_ep, best_vea),
                     xytext=(best_vea_ep, best_vea + 3 if best_vea < 90 else best_vea - 6),
                     arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5),
                     fontweight='bold', fontsize=9, color='#4a148c')

    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='#f8f9fa')

    # 3. Best Accuracy Progress
    ax3 = axs[1, 0]
    ax3.set_title('Cumulative Best Accuracy Progression', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)

    if va_accs:
        best_so_far = []
        cur_max = 0
        for v in va_accs:
            cur_max = max(cur_max, v)
            best_so_far.append(cur_max)
        ax3.plot(va_epochs, best_so_far, 'o-', color='#17becf', linewidth=2.5, markersize=4, label='Cumulative Best Acc@1')

    if vea_accs:
        ema_best_so_far = []
        cur_ema_max = 0
        for v in vea_accs:
            cur_ema_max = max(cur_ema_max, v)
            ema_best_so_far.append(cur_ema_max)
        ax3.plot(vea_epochs, ema_best_so_far, '^--', color='#ff7f0e', linewidth=2, markersize=4, label='Cumulative EMA Best Acc@1')

    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(frameon=True, facecolor='#f8f9fa')

    # 4. Learning Rate & Gradient Norm
    ax4 = axs[1, 1]
    ax4.set_title('Learning Rate & Gradient Norm', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Learning Rate', fontsize=11, color='#1f77b4')

    lr_epochs = sorted(data['train_lrs'].keys())
    lr_vals = [data['train_lrs'][e] for e in lr_epochs]
    l1 = None
    if lr_vals:
        l1 = ax4.plot(lr_epochs, lr_vals, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Learning Rate')
        ax4.tick_params(axis='y', labelcolor='#1f77b4')
        ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    gn_epochs = sorted(data['train_grad_norms'].keys())
    gn_vals = [data['train_grad_norms'][e] for e in gn_epochs]
    l2 = None
    if gn_vals:
        ax4_twin = ax4.twinx()
        ax4_twin.set_ylabel('Avg Grad Norm', fontsize=11, color='#8c564b')
        l2 = ax4_twin.plot(gn_epochs, gn_vals, 's--', color='#8c564b', linewidth=1.5, markersize=4, label='Grad Norm')
        ax4_twin.tick_params(axis='y', labelcolor='#8c564b')

    lines = (l1 or []) + (l2 or [])
    labels = [l.get_label() for l in lines]
    if lines:
        ax4.legend(lines, labels, frameon=True, facecolor='#f8f9fa', loc='upper right')

    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_standalone_curve(x, y_dict, xlabel, ylabel, title, save_path, colors=None, markers=None, linestyles=None):
    """Generates an individual high-resolution plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    default_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf']
    default_markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    default_styles = ['-', '--', '-.', ':']

    for i, (lbl, y) in enumerate(y_dict.items()):
        c = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        m = markers[i] if markers and i < len(markers) else default_markers[i % len(default_markers)]
        ls = linestyles[i] if linestyles and i < len(linestyles) else default_styles[i % len(default_styles)]
        ax.plot(x[:len(y)], y, label=lbl, color=c, marker=m, linestyle=ls, linewidth=2, markersize=5)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#f8f9fa')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_comparison(base_data, cnn_data, out_dir):
    """
    Generates side-by-side and comparative figures between DAMamba Base and DAMamba-CNN Hybrid.
    """
    # 1. Comparison Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("DAMamba Base vs. DAMamba-CNN Hybrid Comparison", fontsize=16, fontweight='bold', y=0.98)

    # Subplot 1: Train Loss Comparison
    ax1 = axs[0, 0]
    ax1.set_title("Training Loss Comparison", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)

    b_te = sorted(base_data['train_losses'].keys())
    b_tl = [base_data['train_losses'][e] for e in b_te]
    c_te = sorted(cnn_data['train_losses'].keys())
    c_tl = [cnn_data['train_losses'][e] for e in c_te]

    ax1.plot(b_te, b_tl, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Base Model Train Loss')
    ax1.plot(c_te, c_tl, 's-', color='#ff7f0e', linewidth=2, markersize=4, label='CNN Hybrid Train Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='#f8f9fa')

    # Subplot 2: Validation Loss Comparison
    ax2 = axs[0, 1]
    ax2.set_title("Validation Loss Comparison", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)

    b_ve = sorted(base_data['val_losses'].keys())
    b_vl = [base_data['val_losses'][e] for e in b_ve]
    c_ve = sorted(cnn_data['val_losses'].keys())
    c_vl = [cnn_data['val_losses'][e] for e in c_ve]

    ax2.plot(b_ve, b_vl, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Base Model Val Loss')
    ax2.plot(c_ve, c_vl, 's-', color='#ff7f0e', linewidth=2, markersize=4, label='CNN Hybrid Val Loss')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='#f8f9fa')

    # Subplot 3: Validation Acc@1 Comparison
    ax3 = axs[1, 0]
    ax3.set_title("Validation Acc@1 (%) Comparison", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Accuracy (%)", fontsize=11)

    b_ae = sorted(base_data['val_acc1'].keys())
    b_acc = [base_data['val_acc1'][e] for e in b_ae]
    c_ae = sorted(cnn_data['val_acc1'].keys())
    c_acc = [cnn_data['val_acc1'][e] for e in c_ae]

    ax3.plot(b_ae, b_acc, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Base Model Val Acc@1')
    ax3.plot(c_ae, c_acc, 's-', color='#ff7f0e', linewidth=2, markersize=4, label='CNN Hybrid Val Acc@1')

    # EMA lines
    b_ema_acc = [base_data['val_ema_acc1'][e] for e in sorted(base_data['val_ema_acc1'].keys())]
    if b_ema_acc:
        ax3.plot(sorted(base_data['val_ema_acc1'].keys()), b_ema_acc, '^--', color='#17becf', linewidth=1.5, markersize=3, label='Base Model EMA Acc@1')
    c_ema_acc = [cnn_data['val_ema_acc1'][e] for e in sorted(cnn_data['val_ema_acc1'].keys())]
    if c_ema_acc:
        ax3.plot(sorted(cnn_data['val_ema_acc1'].keys()), c_ema_acc, 'v--', color='#d62728', linewidth=1.5, markersize=3, label='CNN Hybrid EMA Acc@1')

    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(frameon=True, facecolor='#f8f9fa')

    # Subplot 4: Cumulative Best Accuracy Comparison
    ax4 = axs[1, 1]
    ax4.set_title("Best Acc@1 Progression Comparison", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Accuracy (%)", fontsize=11)

    b_best = []
    cb = 0
    for v in b_acc:
        cb = max(cb, v)
        b_best.append(cb)
    ax4.plot(b_ae, b_best, 'o-', color='#1f77b4', linewidth=2.5, markersize=4, label=f'Base Max: {cb:.2f}%')

    c_best = []
    cc = 0
    for v in c_acc:
        cc = max(cc, v)
        c_best.append(cc)
    ax4.plot(c_ae, c_best, 's-', color='#ff7f0e', linewidth=2.5, markersize=4, label=f'CNN Hybrid Max: {cc:.2f}%')

    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(frameon=True, facecolor='#f8f9fa')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    comp_dash_path = os.path.join(out_dir, "comparison_dashboard.png")
    plt.savefig(comp_dash_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {comp_dash_path}")

    # Comparative standalone plots
    plot_standalone_curve(
        c_te,
        {"Base Model Train Loss": [base_data['train_losses'].get(e, None) for e in c_te if e in base_data['train_losses']],
         "CNN Hybrid Train Loss": c_tl},
        "Epoch", "Loss", "Training Loss: Base Model vs CNN Hybrid",
        os.path.join(out_dir, "comparison_train_loss.png"),
        colors=['#1f77b4', '#ff7f0e']
    )

    plot_standalone_curve(
        c_ve,
        {"Base Model Val Loss": [base_data['val_losses'].get(e, None) for e in c_ve if e in base_data['val_losses']],
         "CNN Hybrid Val Loss": c_vl},
        "Epoch", "Loss", "Validation Loss: Base Model vs CNN Hybrid",
        os.path.join(out_dir, "comparison_val_loss.png"),
        colors=['#1f77b4', '#ff7f0e']
    )

    plot_standalone_curve(
        c_ae,
        {"Base Model Val Acc@1": [base_data['val_acc1'].get(e, None) for e in c_ae if e in base_data['val_acc1']],
         "Base Model EMA Acc@1": [base_data['val_ema_acc1'].get(e, None) for e in c_ae if e in base_data['val_ema_acc1']],
         "CNN Hybrid Val Acc@1": c_acc},
        "Epoch", "Accuracy (%)", "Validation Accuracy: Base Model vs CNN Hybrid",
        os.path.join(out_dir, "comparison_val_accuracy.png"),
        colors=['#1f77b4', '#17becf', '#ff7f0e'],
        linestyles=['-', '--', '-']
    )


def generate_all_plots_for_experiment(data, prefix, out_dir):
    """Generates all individual plots + dashboard for a single experiment."""
    # 1. Dashboard
    dash_path = os.path.join(out_dir, f"{prefix}_dashboard.png")
    plot_single_experiment_dashboard(data, f"Experiment Dashboard: {prefix.replace('_', ' ').title()}", dash_path)

    # Legacy / specific name requested or expected: {prefix}_log.png
    log_png_path = os.path.join(out_dir, f"{prefix}_log.png")
    plot_single_experiment_dashboard(data, f"Training Curves: {prefix.replace('_', ' ').title()}", log_png_path)

    # 2. Train vs Val Loss
    t_epochs = sorted(data['train_losses'].keys())
    t_loss = [data['train_losses'][e] for e in t_epochs]
    v_epochs = sorted(data['val_losses'].keys())
    v_loss = [data['val_losses'][e] for e in v_epochs]

    loss_dict = {'Training Loss': t_loss}
    if v_loss:
        # Align with common epochs or plot directly
        loss_dict['Validation Loss'] = [data['val_losses'].get(e, np.nan) for e in t_epochs]
    if data['val_ema_losses']:
        loss_dict['EMA Validation Loss'] = [data['val_ema_losses'].get(e, np.nan) for e in t_epochs]

    plot_standalone_curve(
        t_epochs, loss_dict,
        "Epoch", "Loss", f"{prefix.replace('_', ' ').title()} - Train vs Validation Loss",
        os.path.join(out_dir, f"{prefix}_train_vs_val_loss.png")
    )

    # 3. Validation Accuracy
    va_epochs = sorted(data['val_acc1'].keys())
    acc_dict = {'Validation Acc@1': [data['val_acc1'][e] for e in va_epochs]}
    if data['val_ema_acc1']:
        acc_dict['EMA Validation Acc@1'] = [data['val_ema_acc1'].get(e, np.nan) for e in va_epochs]

    plot_standalone_curve(
        va_epochs, acc_dict,
        "Epoch", "Accuracy (%)", f"{prefix.replace('_', ' ').title()} - Validation Accuracy",
        os.path.join(out_dir, f"{prefix}_val_accuracy.png"),
        colors=['#2ca02c', '#9467bd'],
        linestyles=['-', '--']
    )

    # 4. Learning Rate Schedule
    lr_epochs = sorted(data['train_lrs'].keys())
    if lr_epochs:
        plot_standalone_curve(
            lr_epochs, {'Learning Rate': [data['train_lrs'][e] for e in lr_epochs]},
            "Epoch", "Learning Rate", f"{prefix.replace('_', ' ').title()} - Learning Rate Schedule",
            os.path.join(out_dir, f"{prefix}_lr_curve.png"),
            colors=['#1f77b4']
        )

    # 5. Gradient Norm
    gn_epochs = sorted(data['train_grad_norms'].keys())
    if gn_epochs:
        plot_standalone_curve(
            gn_epochs, {'Gradient Norm': [data['train_grad_norms'][e] for e in gn_epochs]},
            "Epoch", "Gradient Norm", f"{prefix.replace('_', ' ').title()} - Gradient Norm Progression",
            os.path.join(out_dir, f"{prefix}_grad_norm.png"),
            colors=['#8c564b']
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from log file(s).")
    parser.add_argument('--log', type=str, default=None, help="Path to a single log file")
    parser.add_argument('--out', type=str, default=None, help="Path to save output plot (for single log)")
    parser.add_argument('--logs-dir', type=str, default='logs', help="Directory containing log files")
    parser.add_argument('--all', action='store_true', default=False, help="Process all log files in logs directory and generate comparison")
    args = parser.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    if args.log:
        log_path = os.path.abspath(args.log)
        if not os.path.exists(log_path):
            print(f"Error: Log file {log_path} not found.")
            exit(1)
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        out_path = args.out if args.out else os.path.join(logs_dir, f"{base_name}.png")
        data = parse_logs(log_path)
        plot_single_experiment_dashboard(data, f"Training Curves: {base_name}", out_path)
    else:
        # Default behavior: process both logs in logs/ and generate all individual & comparative plots
        base_log = os.path.join(logs_dir, "damamba_base_model_log.txt")
        cnn_log = os.path.join(logs_dir, "damamba-cnn_log.txt")

        base_data = parse_logs(base_log) if os.path.exists(base_log) else None
        cnn_data = parse_logs(cnn_log) if os.path.exists(cnn_log) else None

        if base_data:
            print("Generating plots for DAMamba Base Model...")
            generate_all_plots_for_experiment(base_data, "damamba_base_model", logs_dir)

        if cnn_data:
            print("Generating plots for DAMamba-CNN Hybrid...")
            generate_all_plots_for_experiment(cnn_data, "damamba_cnn", logs_dir)
            # Also generate damamba-cnn_log.png for exact filename match
            plot_single_experiment_dashboard(cnn_data, "Training Curves: DAMamba-CNN Hybrid", os.path.join(logs_dir, "damamba-cnn_log.png"))

        if base_data and cnn_data:
            print("Generating comparative plots between Base Model and CNN Hybrid...")
            plot_comparison(base_data, cnn_data, logs_dir)

        print("\nAll plots generated successfully in:", logs_dir)
