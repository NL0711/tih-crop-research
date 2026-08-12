"""
evaluate_test.py
================
Full test-set evaluation for DAMamba / CNN+DAMamba.

Never call during training. Only call after training is complete,
or when --eval mode is used.

Usage (standalone CLI)
----------------------
    python evaluate_test.py \
        --cfg configs/DAMamba/damamba_tiny.yaml \
        --resume experiments/.../best_ckpt.pth \
        --data-path /path/to/data \
        --output experiments/...

Usage (from main.py)
--------------------
    from utils.evaluate_test import run_test_evaluation
    run_test_evaluation(config, model, test_loader, class_names, output_dir, tracker)
"""

from __future__ import annotations

import csv, json, logging, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point (called from main.py)
# ---------------------------------------------------------------------------

def run_test_evaluation(
    config,
    model: torch.nn.Module,
    test_loader,
    class_names: List[str],
    output_dir: str,
    tracker=None,
) -> Dict[str, Any]:
    """
    Run full test-set evaluation. Never touches the training loop.

    Returns a dict of scalar metrics.
    """
    if test_loader is None:
        log.warning("[Eval] No test loader – skipping test evaluation.")
        return {}

    log.info("[Eval] Starting test-set evaluation (this does NOT influence training).")
    os.makedirs(output_dir, exist_ok=True)
    pred_dir  = os.path.join(output_dir, "predictions")
    met_dir   = os.path.join(output_dir, "metrics")
    for d in (pred_dir, met_dir): os.makedirs(d, exist_ok=True)

    # 1. Collect predictions --------------------------------------------------
    all_preds, all_probs, all_targets, all_paths, all_times = _collect_predictions(
        model, test_loader, config
    )

    # 2. Save raw predictions CSV ---------------------------------------------
    _save_predictions_csv(
        os.path.join(pred_dir, "test_predictions.csv"),
        all_paths, all_targets, all_preds, all_probs, all_times, class_names
    )
    _save_misclassified_csv(
        os.path.join(pred_dir, "misclassified.csv"),
        all_paths, all_targets, all_preds, all_probs, all_times, class_names
    )

    # 3. Compute metrics ------------------------------------------------------
    metrics = _compute_metrics(all_targets, all_preds, all_probs, class_names)

    # 4. Save metric files ----------------------------------------------------
    _save_classification_report(
        os.path.join(met_dir, "classification_report.csv"),
        metrics["report_dict"], class_names
    )
    _save_per_class_metrics(
        os.path.join(met_dir, "per_class_metrics.csv"),
        metrics["per_class"], class_names
    )
    _save_scalar_metrics(
        os.path.join(output_dir, "test_metrics.json"),
        metrics["scalars"]
    )
    _save_confusion_matrices(
        met_dir, all_targets, all_preds, class_names
    )
    if all_probs is not None:
        _save_roc_pr(met_dir, all_targets, all_probs, class_names)
        _save_calibration(met_dir, all_targets, all_probs, class_names)

    # 5. Inference latency ----------------------------------------------------
    _save_inference_metrics(
        os.path.join(output_dir, "inference_metrics.json"),
        os.path.join(output_dir, "inference_metrics.csv"),
        all_times
    )

    # 6. Update tracker summary -----------------------------------------------
    s = metrics["scalars"]
    test_summary = {
        "test_accuracy":     s.get("top1_accuracy", ""),
        "balanced_accuracy": s.get("balanced_accuracy", ""),
        "macro_f1":          s.get("macro_f1", ""),
        "weighted_f1":       s.get("weighted_f1", ""),
        "macro_precision":   s.get("macro_precision", ""),
        "macro_recall":      s.get("macro_recall", ""),
        "kappa":             s.get("cohen_kappa", ""),
        "mcc":               s.get("matthews_corrcoef", ""),
        "macro_auc":         s.get("macro_roc_auc", ""),
        "ece":               s.get("ece", ""),
        "mean_inference_ms": s.get("mean_inference_ms", ""),
    }
    if tracker is not None:
        try:
            tracker.update_summary_with_test(test_summary)
        except Exception as exc:
            log.warning(f"[Eval] Could not update tracker summary: {exc}")

    log.info(f"[Eval] Test Acc@1 : {s.get('top1_accuracy', 'N/A'):.2f}%")
    log.info(f"[Eval] Macro F1   : {s.get('macro_f1', 'N/A'):.4f}")
    log.info(f"[Eval] Results saved to {output_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def _collect_predictions(model, loader, config):
    model.eval()
    all_preds, all_probs, all_targets, all_paths, all_times = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                images, targets, paths = batch
            else:
                images, targets = batch
                paths = [f"batch{batch_idx}_img{i}" for i in range(len(targets))]

            images  = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            t0 = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                logits = model(images)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(targets)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
            all_targets.extend(targets.cpu().tolist())
            all_paths.extend(list(paths))
            all_times.extend([elapsed_ms] * len(targets))

    all_probs_np = np.concatenate(all_probs, axis=0) if all_probs else None
    return (
        np.array(all_preds),
        all_probs_np,
        np.array(all_targets),
        all_paths,
        np.array(all_times),
    )


# ---------------------------------------------------------------------------
# CSV savers
# ---------------------------------------------------------------------------

def _save_predictions_csv(path, paths, targets, preds, probs, times, class_names):
    correct = (preds == targets)
    with open(path, "w", newline="") as f:
        fields = [
            "image_path", "true_class", "true_class_id",
            "predicted_class", "predicted_class_id",
            "correct", "confidence", "inference_time_ms",
        ]
        if probs is not None:
            fields += [f"prob_{c}" for c in class_names]
            fields += ["top5_classes", "top5_probs"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(targets)):
            row = {
                "image_path":       paths[i],
                "true_class":       class_names[targets[i]] if targets[i] < len(class_names) else str(targets[i]),
                "true_class_id":    int(targets[i]),
                "predicted_class":  class_names[preds[i]]   if preds[i]    < len(class_names) else str(preds[i]),
                "predicted_class_id": int(preds[i]),
                "correct":          bool(correct[i]),
                "confidence":       float(probs[i, preds[i]]) if probs is not None else "",
                "inference_time_ms": float(times[i]),
            }
            if probs is not None:
                for j, c in enumerate(class_names):
                    row[f"prob_{c}"] = float(probs[i, j])
                top5 = np.argsort(probs[i])[::-1][:5]
                row["top5_classes"] = "|".join(class_names[k] if k < len(class_names) else str(k) for k in top5)
                row["top5_probs"]   = "|".join(f"{probs[i,k]:.4f}" for k in top5)
            w.writerow(row)
    log.info(f"[Eval] Saved predictions: {path}")


def _save_misclassified_csv(path, paths, targets, preds, probs, times, class_names):
    wrong = np.where(preds != targets)[0]
    with open(path, "w", newline="") as f:
        fields = [
            "image_path", "true_class", "predicted_class",
            "true_class_id", "predicted_class_id", "confidence",
            "second_best_class", "second_best_prob", "margin", "inference_time_ms",
        ]
        if probs is not None:
            fields += [f"prob_{c}" for c in class_names]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in wrong:
            row = {
                "image_path":       paths[i],
                "true_class":       class_names[targets[i]] if targets[i] < len(class_names) else str(targets[i]),
                "predicted_class":  class_names[preds[i]]   if preds[i]    < len(class_names) else str(preds[i]),
                "true_class_id":    int(targets[i]),
                "predicted_class_id": int(preds[i]),
                "confidence":       float(probs[i, preds[i]]) if probs is not None else "",
                "inference_time_ms": float(times[i]),
                "second_best_class": "",
                "second_best_prob":  "",
                "margin":            "",
            }
            if probs is not None:
                sorted_idx = np.argsort(probs[i])[::-1]
                top1_idx = sorted_idx[0]; top2_idx = sorted_idx[1]
                row["second_best_class"] = class_names[top2_idx] if top2_idx < len(class_names) else str(top2_idx)
                row["second_best_prob"]  = float(probs[i, top2_idx])
                row["margin"]            = float(probs[i, top1_idx] - probs[i, top2_idx])
                for j, c in enumerate(class_names):
                    row[f"prob_{c}"] = float(probs[i, j])
            w.writerow(row)
    log.info(f"[Eval] Saved misclassified: {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(targets, preds, probs, class_names):
    try:
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            precision_score, recall_score, f1_score,
            cohen_kappa_score, matthews_corrcoef,
            classification_report,
            confusion_matrix,
            roc_auc_score, average_precision_score,
            log_loss, brier_score_loss,
        )
        sk_ok = True
    except ImportError:
        log.warning("[Eval] scikit-learn not available – basic metrics only.")
        sk_ok = False

    scalars: Dict[str, Any] = {}
    per_class: Dict[str, Any] = {}
    report_dict: Dict[str, Any] = {}
    n_classes = len(class_names)

    if sk_ok:
        scalars["top1_accuracy"]      = accuracy_score(targets, preds) * 100.0
        scalars["balanced_accuracy"]  = balanced_accuracy_score(targets, preds) * 100.0
        scalars["macro_precision"]    = precision_score(targets, preds, average="macro",  zero_division=0)
        scalars["macro_recall"]       = recall_score(targets,    preds, average="macro",  zero_division=0)
        scalars["macro_f1"]           = f1_score(targets,        preds, average="macro",  zero_division=0)
        scalars["weighted_precision"] = precision_score(targets, preds, average="weighted", zero_division=0)
        scalars["weighted_recall"]    = recall_score(targets,    preds, average="weighted", zero_division=0)
        scalars["weighted_f1"]        = f1_score(targets,        preds, average="weighted", zero_division=0)
        scalars["cohen_kappa"]        = cohen_kappa_score(targets, preds)
        scalars["matthews_corrcoef"]  = matthews_corrcoef(targets, preds)
        report_dict = classification_report(targets, preds,
                                            target_names=class_names,
                                            output_dict=True, zero_division=0)

        # Per-class
        labels = list(range(n_classes))
        per_cls_p = precision_score(targets, preds, labels=labels, average=None, zero_division=0)
        per_cls_r = recall_score(targets,    preds, labels=labels, average=None, zero_division=0)
        per_cls_f = f1_score(targets,        preds, labels=labels, average=None, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=labels)
        for i, cname in enumerate(class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            support = int(cm[i, :].sum())
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            per_class[cname] = {
                "support":     support,
                "correct":     int(tp),
                "incorrect":   int(fn + fp),
                "accuracy":    float(tp / support) if support > 0 else 0.0,
                "precision":   float(per_cls_p[i]),
                "recall":      float(per_cls_r[i]),
                "specificity": float(specificity),
                "f1":          float(per_cls_f[i]),
            }
            if probs is not None:
                # per-class confidence
                idx = np.where(targets == i)[0]
                if len(idx):
                    conf = probs[idx, i]
                    per_class[cname]["mean_confidence"]   = float(np.mean(conf))
                    per_class[cname]["median_confidence"] = float(np.median(conf))

        if probs is not None and n_classes >= 2:
            try:
                scalars["macro_roc_auc"] = roc_auc_score(
                    targets, probs, multi_class="ovr", average="macro"
                )
                scalars["weighted_roc_auc"] = roc_auc_score(
                    targets, probs, multi_class="ovr", average="weighted"
                )
            except Exception as exc:
                log.warning(f"[Eval] ROC-AUC: {exc}")
            try:
                scalars["log_loss"] = log_loss(targets, probs)
            except Exception: pass

            # ECE
            try:
                scalars["ece"] = _compute_ece(targets, probs)
            except Exception: pass

    else:
        correct = (preds == targets).sum()
        scalars["top1_accuracy"] = float(correct) / len(targets) * 100.0

    # Top-5 accuracy (quick numpy version)
    if probs is not None:
        top5_hits = 0
        for i, t in enumerate(targets):
            top5 = np.argsort(probs[i])[::-1][:5]
            if t in top5: top5_hits += 1
        scalars["top5_accuracy"] = top5_hits / len(targets) * 100.0

    return {"scalars": scalars, "per_class": per_class, "report_dict": report_dict}


def _compute_ece(targets, probs, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(float)
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() == 0: continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(targets))


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------

def _save_classification_report(path, report_dict, class_names):
    if not report_dict: return
    rows = []
    for k, v in report_dict.items():
        if isinstance(v, dict):
            row = {"class": k}; row.update(v); rows.append(row)
    if not rows: return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    log.info(f"[Eval] Classification report: {path}")


def _save_per_class_metrics(path, per_class, class_names):
    if not per_class: return
    rows = []
    for cname in class_names:
        if cname in per_class:
            row = {"class": cname}; row.update(per_class[cname]); rows.append(row)
    if not rows: return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    log.info(f"[Eval] Per-class metrics: {path}")


def _save_scalar_metrics(path, scalars):
    with open(path, "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in scalars.items()}, f, indent=2)
    log.info(f"[Eval] Scalar metrics: {path}")


def _save_confusion_matrices(out_dir, targets, preds, class_names):
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = list(range(len(class_names)))
        cm = confusion_matrix(targets, preds, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

        # Save CSVs
        np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm,
                   delimiter=",", fmt="%d",
                   header=",".join(class_names))
        np.savetxt(os.path.join(out_dir, "normalized_confusion_matrix.csv"), cm_norm,
                   delimiter=",", fmt="%.4f",
                   header=",".join(class_names))

        # Save PNGs
        for arr, fname, fmt in [
            (cm,      "confusion_matrix.png",            "d"),
            (cm_norm, "normalized_confusion_matrix.png", ".2f"),
        ]:
            fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
            sns.heatmap(arr, annot=True, fmt=fmt, cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(fname.replace(".png", "").replace("_", " ").title())
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=120)
            plt.close(fig)

        # Top confusion pairs
        pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    pairs.append({
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                        "pct_of_true": float(cm_norm[i, j] * 100),
                    })
        pairs.sort(key=lambda x: -x["count"])
        if pairs:
            with open(os.path.join(out_dir, "top_confusion_pairs.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
                w.writeheader(); w.writerows(pairs[:50])
        log.info(f"[Eval] Confusion matrices saved to {out_dir}")
    except Exception as exc:
        log.warning(f"[Eval] Confusion matrix failed: {exc}")


def _save_roc_pr(out_dir, targets, probs, class_names):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        rows_roc, rows_pr = [], []
        n_classes = len(class_names)
        for i, cname in enumerate(class_names):
            binary = (targets == i).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary): continue
            try:
                roc = roc_auc_score(binary, probs[:, i])
                pr  = average_precision_score(binary, probs[:, i])
                rows_roc.append({"class": cname, "roc_auc": float(roc)})
                rows_pr.append({"class": cname, "pr_auc": float(pr)})
            except Exception: pass

        def _save(rows, path, key):
            if not rows: return
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["class", key])
                w.writeheader(); w.writerows(rows)

        _save(rows_roc, os.path.join(out_dir, "roc_auc_per_class.csv"), "roc_auc")
        _save(rows_pr,  os.path.join(out_dir, "pr_auc_per_class.csv"),  "pr_auc")
        log.info(f"[Eval] ROC/PR AUC saved to {out_dir}")
    except Exception as exc:
        log.warning(f"[Eval] ROC/PR AUC: {exc}")


def _save_calibration(out_dir, targets, probs, class_names, n_bins=15):
    try:
        rows = []
        bins = np.linspace(0, 1, n_bins + 1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == targets).astype(float)
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (confidences >= low) & (confidences < high)
            count = int(mask.sum())
            if count == 0: continue
            rows.append({
                "bin_low": float(low), "bin_high": float(high),
                "count": count,
                "avg_confidence": float(confidences[mask].mean()),
                "avg_accuracy":   float(correct[mask].mean()),
            })
        with open(os.path.join(out_dir, "calibration.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        log.info(f"[Eval] Calibration saved to {out_dir}")
    except Exception as exc:
        log.warning(f"[Eval] Calibration: {exc}")


def _save_inference_metrics(json_path, csv_path, times_ms):
    if len(times_ms) == 0: return
    stats = {
        "count":  len(times_ms),
        "mean":   float(np.mean(times_ms)),
        "median": float(np.median(times_ms)),
        "std":    float(np.std(times_ms)),
        "min":    float(np.min(times_ms)),
        "max":    float(np.max(times_ms)),
        "p90":    float(np.percentile(times_ms, 90)),
        "p95":    float(np.percentile(times_ms, 95)),
        "p99":    float(np.percentile(times_ms, 99)),
    }
    with open(json_path, "w") as f: json.dump(stats, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader(); w.writerow(stats)
    log.info(f"[Eval] Inference metrics: {json_path}")


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from config import get_config
    from models import build_model
    from data import build_loader
    from utils.utils import load_checkpoint_ema
    from utils.logger import create_logger

    parser = argparse.ArgumentParser("Test-set evaluation")
    parser.add_argument("--cfg",        required=True, help="Path to config file")
    parser.add_argument("--resume",     required=True, help="Path to checkpoint")
    parser.add_argument("--data-path",  required=True, help="Dataset root")
    parser.add_argument("--output",     default="./eval_output", help="Output dir")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--opts", nargs="+")
    args = parser.parse_args()
    args.zip         = False
    args.cache_mode  = "no"
    args.pretrained  = None
    args.accumulation_steps = None
    args.use_checkpoint     = False
    args.disable_amp        = False
    args.tag         = "eval"
    args.eval        = True
    args.throughput  = False
    args.oversample  = False
    args.fused_layernorm = False
    args.optim       = None
    args.ddp         = "torch"

    config = get_config(args)
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger(output_dir=args.output, dist_rank=0, name="eval")

    _, _, dataset_test, _, _, data_loader_test, _ = build_loader(config)
    class_names = dataset_test.classes if dataset_test is not None else []

    model = build_model(config)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    from utils.utils import NativeScalerWithGradNormCount
    from utils.lr_scheduler import build_scheduler
    lr_scheduler = build_scheduler(config, optimizer, 1)
    scaler = NativeScalerWithGradNormCount()
    load_checkpoint_ema(config, model, optimizer, lr_scheduler, scaler, logger)

    metrics = run_test_evaluation(
        config, model, data_loader_test, class_names, args.output, tracker=None
    )
    print(json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in metrics.get("scalars", {}).items()},
        indent=2
    ))
