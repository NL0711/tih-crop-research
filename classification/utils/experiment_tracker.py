from __future__ import annotations
import csv, json, logging, os, platform, shutil, subprocess, sys, time, datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

def _is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def _mount_drive(mount_point="/content/drive"):
    try:
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        log.info(f"Google Drive mounted at {mount_point}")
        return True
    except Exception as exc:
        log.warning(f"Could not mount Google Drive: {exc}")
        return False

def _get_git_info():
    info = {"commit": "N/A", "branch": "N/A"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    return info

def _get_env_str():
    import torch
    lines = [
        f"Python:      {sys.version}",
        f"Platform:    {platform.platform()}",
        f"PyTorch:     {torch.__version__}",
        f"CUDA avail:  {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines += [
            f"CUDA ver:    {torch.version.cuda}",
            f"GPU name:    {torch.cuda.get_device_name(0)}",
        ]
    try:
        import torchvision; lines.append(f"torchvision: {torchvision.__version__}")
    except Exception: pass
    try:
        import timm; lines.append(f"timm:        {timm.__version__}")
    except Exception: pass
    git = _get_git_info()
    lines += [f"git commit:  {git['commit']}", f"git branch:  {git['branch']}"]
    return "\n".join(lines)

def _safe_plot(x, y_dict, xlabel, ylabel, title, save_path, legend=True):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl, y in y_dict.items():
            ax.plot(x[:len(y)], y, label=lbl, linewidth=1.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        if legend: ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    except Exception as exc:
        log.warning(f"Cannot save plot {save_path}: {exc}")

def _append_csv_row(path, row):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if write_header: w.writeheader()
            w.writerow(row)
    except Exception as exc:
        log.warning(f"Cannot append to {path}: {exc}")

def _mirror_dir(src, dst):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dest_root = os.path.join(dst, rel)
        os.makedirs(dest_root, exist_ok=True)
        for fname in files:
            sf = os.path.join(root, fname)
            df = os.path.join(dest_root, fname)
            if not os.path.exists(df) or os.path.getmtime(sf) > os.path.getmtime(df):
                shutil.copy2(sf, df)


class ExperimentTracker:
    def __init__(self, config, args, dataset_info=None):
        self.config = config
        self.args = args
        self.dataset_info = dataset_info or {}
        self.in_colab = _is_colab()
        exp_cfg = getattr(config, "EXPERIMENT", None)
        self.drive_root = (getattr(exp_cfg, "DRIVE_ROOT", None) or
                           "/content/drive/MyDrive/DAMamba_Experiments")
        self.local_root = getattr(exp_cfg, "LOCAL_ROOT", None) or "./experiments"
        self.enable_drive = getattr(exp_cfg, "ENABLE_DRIVE", True) if exp_cfg is not None else True
        hybrid = False
        try: hybrid = config.MODEL.HYBRID.ENABLE
        except Exception: pass
        self.arch_tag = "HYBRID" if hybrid else "VANILLA"
        self.model_name = config.MODEL.NAME
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_name = f"{self.model_name}_{self.arch_tag}"
        self.exp_dir = None
        self.drive_exp_dir = None
        self._epoch_rows = []
        self._best_val_acc = 0.0
        self._best_epoch = 0
        self._training_start = 0.0
        self._csv_fields = None

    def start(self):
        self._setup_dirs()
        self._save_startup()
        log.info(f"[Tracker] Experiment dir : {self.exp_dir}")
        if self.drive_exp_dir:
            log.info(f"[Tracker] Drive mirror   : {self.drive_exp_dir}")
        self._training_start = time.time()

    def log_epoch(self, epoch, metrics):
        row = {"epoch": epoch, "timestamp": datetime.datetime.now().isoformat()}
        row.update(metrics)
        self._epoch_rows.append(row)
        self._flush_csv(row)
        self._flush_json()
        for p in (self._csv_path(), self._json_path()):
            self._sync_file(p)

    def on_best_model(self, epoch, acc1):
        self._best_val_acc = acc1
        self._best_epoch = epoch
        log.info(f"[Tracker] New best – epoch {epoch}, acc1={acc1:.2f}%")
        self._sync_dir()

    def on_training_complete(self, total_seconds):
        log.info("[Tracker] Saving plots and final report ...")
        self._save_plots()
        self._append_summary(total_seconds)
        self._generate_report(total_seconds)
        self._sync_dir()
        log.info(f"[Tracker] Done. Artefacts: {self.exp_dir}")
        if self.drive_exp_dir:
            log.info(f"[Tracker] Drive copy: {self.drive_exp_dir}")

    def update_summary_with_test(self, test_metrics):
        for path in (
            os.path.join(self.exp_dir, "comparison", "experiment_summary.csv"),
            os.path.join(self.local_root, "experiment_summary.csv"),
        ):
            if not os.path.exists(path): continue
            try:
                import pandas as pd
                df = pd.read_csv(path)
                mask = df["timestamp"] == self.timestamp
                for k, v in test_metrics.items():
                    if k in df.columns: df.loc[mask, k] = v
                df.to_csv(path, index=False)
            except Exception as exc:
                log.warning(f"[Tracker] Could not patch summary: {exc}")

    def set_dataset_info(self, info):
        self.dataset_info.update(info)

    def subdir(self, name):
        d = os.path.join(self.exp_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def sync_checkpoint_to_drive(self, ckpt_path):
        self._sync_file(ckpt_path, subdir="checkpoints")

    # ---------- private ----------

    def _setup_dirs(self):
        base = os.path.join(self.local_root, self.exp_name, self.timestamp)
        os.makedirs(base, exist_ok=False)
        self.exp_dir = base
        for sub in ("predictions", "metrics", "plots", "comparison"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        if self.in_colab and self.enable_drive:
            if _mount_drive():
                db = os.path.join(self.drive_root, self.exp_name, self.timestamp)
                try:
                    for sub in ("predictions", "metrics", "plots", "comparison", "checkpoints"):
                        os.makedirs(os.path.join(db, sub), exist_ok=True)
                    self.drive_exp_dir = db
                except Exception as exc:
                    log.warning(f"[Tracker] Drive dir: {exc}")

    def _save_startup(self):
        with open(os.path.join(self.exp_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        try:
            with open(os.path.join(self.exp_dir, "environment.txt"), "w") as f:
                f.write(_get_env_str() + "\n")
        except Exception as exc:
            log.warning(f"[Tracker] env info: {exc}")
        try:
            with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
                f.write(self.config.dump())
        except Exception as exc:
            log.warning(f"[Tracker] config.yaml: {exc}")
        meta = self._build_meta()
        with open(os.path.join(self.exp_dir, "experiment_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        for fn in ("command.txt", "environment.txt", "config.yaml", "experiment_metadata.json"):
            self._sync_file(os.path.join(self.exp_dir, fn))

    def _build_meta(self):
        import torch
        cfg = self.config; git = _get_git_info()
        hybrid = False
        try: hybrid = cfg.MODEL.HYBRID.ENABLE
        except Exception: pass
        meta = {
            "experiment_name": self.exp_name, "timestamp": self.timestamp,
            "architecture": cfg.MODEL.TYPE, "model_name": cfg.MODEL.NAME,
            "hybrid_enabled": hybrid, "num_classes": cfg.MODEL.NUM_CLASSES,
            "dataset_path": cfg.DATA.DATA_PATH, "batch_size": cfg.DATA.BATCH_SIZE,
            "optimizer": cfg.TRAIN.OPTIMIZER.NAME, "base_lr": cfg.TRAIN.BASE_LR,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY, "scheduler": cfg.TRAIN.LR_SCHEDULER.NAME,
            "warmup_epochs": cfg.TRAIN.WARMUP_EPOCHS, "epochs": cfg.TRAIN.EPOCHS,
            "seed": cfg.SEED, "amp_enabled": cfg.AMP_ENABLE,
            "label_smoothing": cfg.MODEL.LABEL_SMOOTHING,
            "pretrained": cfg.MODEL.PRETRAINED, "resume": cfg.MODEL.RESUME,
            "git_commit": git["commit"], "git_branch": git["branch"],
            "python_version": sys.version, "platform": platform.platform(),
        }
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["cuda_version"] = torch.version.cuda
        meta.update(self.dataset_info)
        return meta

    def _csv_path(self): return os.path.join(self.exp_dir, "metrics.csv")
    def _json_path(self): return os.path.join(self.exp_dir, "metrics.json")

    def _flush_csv(self, row):
        path = self._csv_path()
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
        else:
            for k in row:
                if k not in self._csv_fields: self._csv_fields.append(k)
        mode = "a" if os.path.exists(path) else "w"
        with open(path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            if mode == "w": w.writeheader()
            w.writerow(row)

    def _flush_json(self):
        with open(self._json_path(), "w") as f:
            json.dump(self._epoch_rows, f, indent=2, default=str)

    def _save_plots(self):
        if not self._epoch_rows: return
        pdir = os.path.join(self.exp_dir, "plots")
        epochs = [r["epoch"] for r in self._epoch_rows]

        def series(k): return [r.get(k) for r in self._epoch_rows if r.get(k) is not None]
        def ep_for(k): return [r["epoch"] for r in self._epoch_rows if r.get(k) is not None]

        for key, lbl, fname in [
            ("train_loss","Train Loss","train_loss.png"),
            ("val_loss","Val Loss","val_loss.png"),
            ("val_acc1","Val Acc@1 (%)","val_accuracy.png"),
            ("train_acc1","Train Acc@1 (%)","train_accuracy.png"),
            ("lr","Learning Rate","lr_curve.png"),
            ("grad_norm","Gradient Norm","grad_norm.png"),
        ]:
            s = series(key)
            if s: _safe_plot(ep_for(key), {lbl: s}, "Epoch", lbl, lbl,
                             os.path.join(pdir, fname), legend=False)
        tl = series("train_loss"); vl = series("val_loss")
        if tl or vl:
            d = {}
            if tl: d["Train"] = tl
            if vl: d["Val"] = vl
            _safe_plot(epochs, d, "Epoch", "Loss", "Train vs Val Loss",
                       os.path.join(pdir, "train_vs_val_loss.png"))
        ta = series("train_acc1"); va = series("val_acc1")
        if ta or va:
            d = {}
            if ta: d["Train"] = ta
            if va: d["Val"] = va
            _safe_plot(epochs, d, "Epoch", "Accuracy (%)", "Train vs Val Accuracy",
                       os.path.join(pdir, "train_vs_val_accuracy.png"))
        for fn in os.listdir(pdir):
            self._sync_file(os.path.join(pdir, fn), subdir="plots")

    def _append_summary(self, total_seconds):
        last = self._epoch_rows[-1] if self._epoch_rows else {}
        try:
            hybrid_val = self.config.MODEL.HYBRID.ENABLE
        except Exception:
            hybrid_val = False
        row = {
            "experiment_name": self.exp_name, "timestamp": self.timestamp,
            "architecture": self.config.MODEL.TYPE, "hybrid_enabled": hybrid_val,
            "best_val_accuracy": self._best_val_acc, "best_epoch": self._best_epoch,
            "final_val_accuracy": last.get("val_acc1", ""),
            "test_accuracy": "", "balanced_accuracy": "", "macro_f1": "",
            "weighted_f1": "", "macro_precision": "", "macro_recall": "",
            "kappa": "", "mcc": "", "macro_auc": "", "ece": "",
            "parameter_count": self.dataset_info.get("parameter_count", ""),
            "training_time_s": round(total_seconds, 1), "mean_inference_ms": "",
            "exp_dir": self.exp_dir,
        }
        for path in (
            os.path.join(self.exp_dir, "comparison", "experiment_summary.csv"),
            os.path.join(self.local_root, "experiment_summary.csv"),
        ):
            _append_csv_row(path, row)
        if self.drive_exp_dir:
            _append_csv_row(os.path.join(self.drive_root, "experiment_summary.csv"), row)

    def _generate_report(self, total_seconds):
        path = os.path.join(self.exp_dir, "FINAL_REPORT.md")
        elapsed = str(datetime.timedelta(seconds=int(total_seconds)))
        cfg = self.config; di = self.dataset_info; git = _get_git_info()
        hybrid = False
        try: hybrid = cfg.MODEL.HYBRID.ENABLE
        except Exception: pass
        last = self._epoch_rows[-1] if self._epoch_rows else {}
        lines = [
            "# DAMamba Experiment – Final Report", "",
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "", "---", "", "## Experiment", "",
            "| Key | Value |", "|-----|-------|",
            f"| Name | {self.exp_name} |",
            f"| Timestamp | {self.timestamp} |",
            f"| Architecture | {cfg.MODEL.TYPE} |",
            f"| Model | {cfg.MODEL.NAME} |",
            f"| Hybrid enabled | {hybrid} |",
            f"| Output dir | {self.exp_dir} |",
            "", "## Dataset", "",
            "| Split | Samples |", "|-------|---------|",
            f"| Train | {di.get('train_size', 'N/A')} |",
            f"| Val   | {di.get('val_size', 'N/A')} |",
            f"| Test  | {di.get('test_size', 'N/A')} |",
            f"| Classes | {di.get('num_classes', cfg.MODEL.NUM_CLASSES)} |",
            "", "## Training Configuration", "",
            "| Param | Value |", "|-------|-------|",
            f"| Epochs | {cfg.TRAIN.EPOCHS} |",
            f"| Batch size | {cfg.DATA.BATCH_SIZE} |",
            f"| Base LR | {cfg.TRAIN.BASE_LR} |",
            f"| Weight decay | {cfg.TRAIN.WEIGHT_DECAY} |",
            f"| Optimizer | {cfg.TRAIN.OPTIMIZER.NAME} |",
            f"| Scheduler | {cfg.TRAIN.LR_SCHEDULER.NAME} |",
            f"| Warmup epochs | {cfg.TRAIN.WARMUP_EPOCHS} |",
            f"| AMP | {cfg.AMP_ENABLE} |",
            f"| Seed | {cfg.SEED} |",
            f"| Total training time | {elapsed} |",
            "", "## Validation", "",
            "| Metric | Value |", "|--------|-------|",
            f"| Best Val Acc@1 | **{self._best_val_acc:.2f}%** |",
            f"| Best Epoch | {self._best_epoch} |",
            f"| Final Val Acc@1 | {last.get('val_acc1', 'N/A')} |",
            f"| Final Val Acc@5 | {last.get('val_acc5', 'N/A')} |",
            "", "## Test", "",
            "_Run evaluate_test.py or use --eval flag to populate test metrics._",
            "", "## Plots", "", "See plots/ subdirectory.", "",
            "## Reproducibility", "",
            "| Item | Value |", "|------|-------|",
            f"| Command | {' '.join(sys.argv)} |",
            f"| Git commit | {git['commit']} |",
            f"| Git branch | {git['branch']} |",
            f"| Seed | {cfg.SEED} |",
            f"| Config | {os.path.join(self.exp_dir, 'config.yaml')} |",
            "",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        log.info(f"[Tracker] Final report: {path}")
        self._sync_file(path)

    def _sync_file(self, local_path, subdir=""):
        if not self.drive_exp_dir or not os.path.isfile(local_path): return
        try:
            dest = os.path.join(self.drive_exp_dir, subdir) if subdir else self.drive_exp_dir
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(local_path, dest)
        except Exception as exc:
            log.warning(f"[Tracker] Drive sync failed {local_path}: {exc}")

    def _sync_dir(self):
        if not self.drive_exp_dir: return
        try: _mirror_dir(self.exp_dir, self.drive_exp_dir)
        except Exception as exc: log.warning(f"[Tracker] Dir sync failed: {exc}")
