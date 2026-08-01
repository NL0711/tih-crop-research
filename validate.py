#!/usr/bin/env python3
"""
Model Validation Script for DAMamba Classification Models.
This script evaluates a trained checkpoint (e.g., best_ckpt.pth) on a dataset split,
calculates overall and class-wise metrics, and generates a visual report including
a confusion matrix.
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

# Add classification folder and root folder to python path to ensure imports work from anywhere
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'classification'))

try:
    from classification.config import get_config
    from classification.models import build_model
    from classification.data.build import build_transform
except ImportError as e:
    print(f"Error importing repository modules: {e}")
    print("Please make sure you run this script from the project root directory.")
    sys.path.append(os.path.join(root_dir, 'classification'))
    from config import get_config
    from models import build_model
    from data.build import build_transform


# Custom ImageFolder that enforces a predefined class-to-index mapping
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, custom_classes=None, **kwargs):
        self.custom_classes = custom_classes
        super().__init__(root, transform=transform, **kwargs)

    def find_classes(self, directory):
        if self.custom_classes is not None:
            classes = self.custom_classes
            class_to_idx = {c: i for i, c in enumerate(classes)}
            return classes, class_to_idx
        return super().find_classes(directory)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DAMamba classification checkpoint on a dataset.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=os.path.join(root_dir, "output", "tiny", "finetune", "best_ckpt.pth"),
        help="Path to the model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--cfg", 
        type=str, 
        default="", 
        help="Path to model config file (.yaml or .json). If empty, auto-detects config.json next to checkpoint."
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default="", 
        help="Path to dataset directory. If empty, uses the path specified in the config."
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for validation."
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=0, 
        help="Number of workers for data loading."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run validation on (cuda or cpu)."
    )
    parser.add_argument(
        "--use-ema", 
        action="store_true", 
        help="Use EMA weights from the checkpoint if available."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=os.path.join(root_dir, "output", "tiny", "finetune", "validation_results"),
        help="Directory to save validation results (plots and json report)."
    )
    
    # These match options from main.py config mapping
    parser.add_argument('--opts', help="Modify config options", default=None, nargs='+')
    
    # Return dummy arg objects to satisfy config parser requirements
    args = parser.parse_args()
    return args


def compute_metrics(y_true, y_pred, classes):
    """
    Computes accuracy, class-wise precision, recall, f1-score, and confusion matrix.
    """
    num_classes = len(classes)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Class-wise metrics
    class_metrics = {}
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
        
    for idx, class_name in enumerate(classes):
        tp = confusion_matrix[idx, idx]
        fp = np.sum(confusion_matrix[:, idx]) - tp
        fn = np.sum(confusion_matrix[idx, :]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(y_true == idx))
        
        class_metrics[class_name] = {
            "index": idx,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }
        
    # Macro averages
    macro_precision = np.mean([m["precision"] for m in class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in class_metrics.values()])
    macro_f1 = np.mean([m["f1_score"] for m in class_metrics.values()])
    
    # Weighted averages
    total_support = len(y_true)
    weighted_precision = np.sum([m["precision"] * m["support"] for m in class_metrics.values()]) / total_support if total_support > 0 else 0.0
    weighted_recall = np.sum([m["recall"] * m["support"] for m in class_metrics.values()]) / total_support if total_support > 0 else 0.0
    weighted_f1 = np.sum([m["f1_score"] * m["support"] for m in class_metrics.values()]) / total_support if total_support > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1)
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1)
        },
        "class_wise": class_metrics,
        "confusion_matrix": confusion_matrix.tolist()
    }
    
    return metrics


def print_metrics_table(metrics, classes):
    """
    Prints a beautiful performance table.
    """
    print("\n" + "="*80)
    print(f"{'CLASS-WISE PERFORMANCE REPORT':^80}")
    print("="*80)
    print(f" {'Class Name':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8} ")
    print("-"*80)
    
    for class_name in classes:
        m = metrics["class_wise"][class_name]
        print(f" {class_name:<30} | {m['precision']:10.4f} | {m['recall']:10.4f} | {m['f1_score']:10.4f} | {m['support']:<8} ")
        
    print("-"*80)
    print(f" {'Accuracy':<30} | {'':<10} | {'':<10} | {metrics['accuracy']:10.4f} | {sum(m['support'] for m in metrics['class_wise'].values()):<8} ")
    print(f" {'Macro Average':<30} | {metrics['macro_avg']['precision']:10.4f} | {metrics['macro_avg']['recall']:10.4f} | {metrics['macro_avg']['f1_score']:10.4f} |")
    print(f" {'Weighted Average':<30} | {metrics['weighted_avg']['precision']:10.4f} | {metrics['weighted_avg']['recall']:10.4f} | {metrics['weighted_avg']['f1_score']:10.4f} |")
    print("="*80 + "\n")


def plot_and_save_visuals(metrics, classes, output_dir):
    """
    Plots and saves confusion matrix and class F1-scores if matplotlib/seaborn are available.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot Confusion Matrix
        cm = np.array(metrics["confusion_matrix"])
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix for color map visualization but show raw counts
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm) # Replace NaN if support is 0
        
        sns.heatmap(
            cm_norm, 
            annot=cm, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=classes, 
            yticklabels=classes,
            cbar=True,
            square=True,
            annot_kws={"size": 12}
        )
        plt.title("Confusion Matrix (Raw Counts / Row-Normalized Heatmap)", fontsize=14, pad=15)
        plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
        plt.ylabel("True Labels", fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to: {cm_path}")
        
        # Plot F1-Score Bar Chart
        plt.figure(figsize=(10, 6))
        f1_scores = [metrics["class_wise"][c]["f1_score"] for c in classes]
        colors = sns.color_palette("viridis", len(classes))
        
        bars = plt.bar(classes, f1_scores, color=colors, edgecolor='grey', width=0.6)
        plt.title("Class-wise F1-Scores", fontsize=14, pad=15)
        plt.xlabel("Classes", fontsize=12, labelpad=10)
        plt.ylabel("F1-Score", fontsize=12, labelpad=10)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.0, 
                height + 0.02, 
                f"{height:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=10,
                weight='bold'
            )
            
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        
        f1_path = os.path.join(output_dir, "f1_scores.png")
        plt.savefig(f1_path, dpi=300)
        plt.close()
        print(f"Saved F1-scores performance plot to: {f1_path}")
        
    except ImportError:
        print("Matplotlib or Seaborn not installed. Skipping visualization plots.")


def main():
    args = parse_args()
    
    # 1. Check and resolve checkpoint path
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist.")
        sys.exit(1)
        
    # 2. Auto-detect config.json (YAML format) next to checkpoint if not specified
    config_file = args.cfg
    temp_yaml_to_clean = None
    if not config_file:
        ckpt_dir = os.path.dirname(checkpoint_path)
        possible_config = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(possible_config):
            temp_yaml = os.path.join(ckpt_dir, "temp_config_yacs.yaml")
            try:
                import shutil
                shutil.copyfile(possible_config, temp_yaml)
                config_file = temp_yaml
                temp_yaml_to_clean = temp_yaml
                print(f"Auto-detected model configuration at: {possible_config} (copied to temp_config_yacs.yaml)")
            except Exception as e:
                print(f"Warning: Could not create temporary YAML config copy: {e}. Trying direct load...")
                config_file = possible_config
        else:
            # Fall back to a default config inside configs directory
            default_config = os.path.join(root_dir, "classification", "configs", "DAMamba", "damamba_tiny.yaml")
            if os.path.exists(default_config):
                config_file = default_config
                print(f"No config.json found next to checkpoint. Falling back to default: {config_file}")
            else:
                print("Error: Could not locate a valid model configuration file. Please provide one using --cfg.")
                sys.exit(1)
                
    # 3. Load model configuration node using CfgNode (reusing repository code)
    print("Loading configuration...")
    # Construct a temporary namespace to satisfy get_config's interface
    class TempArgs:
        def __init__(self, cfg, opts=None):
            self.cfg = cfg
            self.opts = opts
            self.batch_size = args.batch_size
            self.data_path = args.dataset_path if args.dataset_path else None
            self.zip = False
            self.cache_mode = None
            self.pretrained = None
            self.resume = checkpoint_path
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.disable_amp = False
            self.output = None
            self.tag = None
            self.oversample = None
            self.eval = True
            self.throughput = False
            self.traincost = False
            self.enable_persistance = False
            self.enable_amp = False
            self.fused_layernorm = False
            self.optim = None
            self.ddp = 'torch'
            
    temp_args = TempArgs(config_file, args.opts)
    try:
        config = get_config(temp_args)
    finally:
        if temp_yaml_to_clean and os.path.exists(temp_yaml_to_clean):
            try:
                os.remove(temp_yaml_to_clean)
            except Exception:
                pass
    
    # Determine the target dataset split directory
    dataset_base = args.dataset_path if args.dataset_path else config.DATA.DATA_PATH
    if not dataset_base:
        print("Error: Dataset path is empty. Please specify it using --dataset-path.")
        sys.exit(1)
        
    if not os.path.isabs(dataset_base):
        dataset_base = os.path.abspath(os.path.join(root_dir, dataset_base))
        
    split_dir = os.path.join(dataset_base, args.split)
    if os.path.isdir(split_dir):
        target_dir = split_dir
    elif os.path.isdir(dataset_base):
        target_dir = dataset_base
    else:
        print(f"Error: Dataset directory '{dataset_base}' (or its '{args.split}' split) not found.")
        sys.exit(1)
        
    print(f"Validation Target Directory: {target_dir}")
    
    # 4. Determine classes and check model head shape
    print("Reading checkpoint weight keys...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to find model state dict key
    if 'model' in checkpoint:
        state_key = 'model'
    elif 'model_ema' in checkpoint:
        state_key = 'model_ema'
    else:
        # Checkpoint is just the raw state dict
        state_key = None
        state_dict = checkpoint
        
    if state_key is not None:
        state_dict = checkpoint[state_key]
        
    # Infer number of classes from final linear classification layer in the checkpoint
    # DAMamba classification head is MlpHead, containing self.fc = nn.Linear(dim, num_classes)
    head_weight_key = None
    for k in state_dict.keys():
        if 'head.fc.weight' in k:
            head_weight_key = k
            break
            
    if head_weight_key is not None:
        num_classes_checkpoint = state_dict[head_weight_key].shape[0]
        print(f"Detected {num_classes_checkpoint} output classes in model checkpoint.")
    else:
        num_classes_checkpoint = config.MODEL.NUM_CLASSES
        print(f"Could not identify classification head in checkpoint. Defaulting to config classes: {num_classes_checkpoint}")
        
    # Enforce checkpoint number of classes in config
    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes_checkpoint
    config.freeze()
    
    # 5. Build Class mapping to prevent label indexing mismatches
    detected_classes = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    print(f"Detected classes in target directory: {detected_classes}")
    
    custom_classes = None
    if len(detected_classes) != num_classes_checkpoint:
        print(f"Warning: Target directory has {len(detected_classes)} class folders, but the model has {num_classes_checkpoint} outputs.")
        
    # Look for split_summary.json or sibling folders to get the exact class ordering
    parent_dir = os.path.dirname(target_dir)
    sibling_folders = ['train', 'val', 'test']
    
    for sib in sibling_folders:
        sib_path = os.path.join(parent_dir, sib)
        if os.path.isdir(sib_path):
            sib_classes = sorted([d for d in os.listdir(sib_path) if os.path.isdir(os.path.join(sib_path, d))])
            if len(sib_classes) == num_classes_checkpoint:
                custom_classes = sib_classes
                print(f"Enforced class list from sibling folder '{sib}': {custom_classes}")
                break
                
    if custom_classes is None:
        # Check split_summary.json in parent or grandparent folder
        possible_json = [
            os.path.join(parent_dir, "split_summary.json"),
            os.path.join(os.path.dirname(parent_dir), "split_summary.json")
        ]
        for json_path in possible_json:
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                    if 'classes' in meta:
                        json_classes = sorted(list(meta['classes'].keys()))
                        if len(json_classes) == num_classes_checkpoint:
                            custom_classes = json_classes
                            print(f"Enforced class list from split_summary.json: {custom_classes}")
                            break
                except Exception:
                    pass
                    
    if custom_classes is None:
        if len(detected_classes) == num_classes_checkpoint:
            custom_classes = detected_classes
        else:
            print("Error: Class count mismatch between checkpoint and dataset. Cannot map class indices reliably.")
            print("Please make sure the dataset split structure matches the training configuration.")
            sys.exit(1)
            
    # 6. Build model
    print(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    
    # 7. Load checkpoint weights into model
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Choose weights (EMA vs Student)
    if args.use_ema:
        if 'model_ema' in checkpoint:
            print("Loading model_ema weights...")
            weights = checkpoint['model_ema']
        else:
            print("Warning: Model EMA not found in checkpoint. Falling back to student model weights.")
            weights = checkpoint.get('model', checkpoint)
    else:
        print("Loading student model weights...")
        weights = checkpoint.get('model', checkpoint)
        
    # Remove 'module.' prefix if saved in distributed mode
    cleaned_weights = {}
    for k, v in weights.items():
        key_name = k[7:] if k.startswith('module.') else k
        cleaned_weights[key_name] = v
        
    msg = model.load_state_dict(cleaned_weights, strict=True)
    print(f"Loaded weights successfully: {msg}")
    
    model = model.to(device)
    model.eval()
    
    # 8. Build Dataset and DataLoader
    print("Preparing dataset transforms and loaders...")
    transform = build_transform(is_train=False, config=config)
    dataset = CustomImageFolder(target_dir, transform=transform, custom_classes=custom_classes)
    
    print(f"Evaluation dataset size: {len(dataset)} samples")
    print(f"Class-to-index mapping being used: {dataset.class_to_idx}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    
    # 9. Validation loop
    y_true = []
    y_pred = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Use PyTorch mixed precision (AMP) if enabled in config
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(images)
                
            predictions = torch.argmax(outputs, dim=-1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            
    total_time = time.time() - start_time
    fps = len(dataset) / total_time
    avg_latency_ms = (total_time / len(dataset)) * 1000
    
    print(f"\nInference completed in {total_time:.2f} seconds.")
    print(f"Throughput: {fps:.2f} images/sec (FPS)")
    print(f"Average latency per image: {avg_latency_ms:.2f} ms")
    
    # 10. Compute and Save Metrics
    metrics = compute_metrics(y_true, y_pred, custom_classes)
    print_metrics_table(metrics, custom_classes)
    
    # Save text report and json metrics
    os.makedirs(args.output_dir, exist_ok=True)
    
    report_json_path = os.path.join(args.output_dir, "validation_metrics.json")
    with open(report_json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved detailed report JSON to: {report_json_path}")
    
    # Plot visuals
    plot_and_save_visuals(metrics, custom_classes, args.output_dir)
    print(f"\nAll validation results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
