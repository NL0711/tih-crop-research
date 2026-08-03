import os
import sys
import argparse
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

# Add classification folder to path to resolve imports properly
sys.path.append(os.path.join(os.path.dirname(__file__), 'classification'))

from classification.models import build_model
from classification.config import get_config
from classification.debug.visualizer import DASVisualizer
from classification.debug.export import export_debug_data

def preprocess_image(img_path, img_size=224):
    """
    Loads and preprocesses an image for DAMamba input.
    """
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Preprocessing transforms (Resize, normalize)
    img_resized = img.resize((img_size, img_size), Image.BICUBIC)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_np - mean) / std
    
    # HWC to CHW and batch dimension
    tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, np.array(img), (orig_h, orig_w)

def load_checkpoint(model, ckpt_path):
    """
    Loads checkpoint state_dict into the model.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Clean up keys if needed (e.g. remove 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Checkpoint loaded: {msg}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Dynamic Adaptive Scan in DAMamba")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="classification/configs/DAMamba/damamba_tiny.yaml", help="Path to config yaml")
    parser.add_argument("--images", type=str, required=True, help="Directory containing images to evaluate")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save output files")
    parser.add_argument("--true-labels", type=str, default=None, help="JSON file mapping image file names to class index (optional)")
    args = parser.parse_args()
    
    # Load configuration
    from yacs.config import CfgNode as CN
    # Create a dummy args-like class for yacs config updater
    class DummyArgs:
        def __init__(self, cfg_file):
            self.cfg = cfg_file
            self.opts = []
            self.batch_size = None
            self.data_path = None
            self.cache_mode = None
            self.pretrained = None
            self.resume = None
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.disable_amp = False
            self.output = "outputs"
            self.tag = "eval"
            self.oversample = False
            
    config = get_config(DummyArgs(args.config))
    
    # Instantiate the model
    model = build_model(config)
    if model is None:
        raise ValueError(f"Failed to build model from config: {args.config}")
        
    # Load weights
    load_checkpoint(model, args.model)
    model.eval()
    
    # Load true labels if available
    true_labels = {}
    if args.true_labels and os.path.exists(args.true_labels):
        with open(args.true_labels, 'r') as f:
            true_labels = json.load(f)
            
    # Gather images
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.JPEG', '.PNG', '.JPG')
    image_files = [
        f for f in os.listdir(args.images) 
        if os.path.isfile(os.path.join(args.images, f)) and f.endswith(image_exts)
    ]
    
    if not image_files:
        print(f"No images found in {args.images}")
        return
        
    print(f"Found {len(image_files)} images. Starting inference and scan analysis...")
    os.makedirs(args.output, exist_ok=True)
    
    csv_rows = []
    
    # Enable instrumentation
    model.enable_scan_debug()
    
    visualizer = DASVisualizer(args.output)
    
    for img_file in image_files:
        img_path = os.path.join(args.images, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        # Preprocess
        tensor, raw_image, orig_size = preprocess_image(img_path, config.DATA.IMG_SIZE)
        
        # Forward pass
        model.clear_scan_debug()
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, dim=0)
            
        pred_class = int(predicted.item())
        conf_val = float(confidence.item())
        
        # Fetch true label if available
        gt_class = true_labels.get(img_file, true_labels.get(img_name, None))
        is_correct = True
        if gt_class is not None:
            is_correct = (pred_class == int(gt_class))
            
        # Get recorded debug records
        debug_records = model.get_scan_debug_data()
        
        # Save visualization and arrays
        meta_extra = {
            "predicted_class": pred_class,
            "confidence": conf_val,
            "true_label": gt_class,
            "correct": is_correct,
            "input_image_size": orig_size
        }
        
        export_debug_data(args.output, img_name, debug_records, meta_extra)
        visualizer.save(raw_image, img_name, debug_records)
        
        # Collect statistics per stage/block for the CSV
        for record in debug_records:
            metrics = record["offset_metrics"]
            feat_metrics = record["feature_metrics"]
            csv_rows.append({
                "image_name": img_file,
                "predicted_class": pred_class,
                "confidence": conf_val,
                "true_label": gt_class if gt_class is not None else "N/A",
                "correct": is_correct,
                "stage": record["stage"],
                "block": record["block"],
                "resolution": f"{record['H']}x{record['W']}",
                "offset_mean": metrics["offset_mean"],
                "offset_std": metrics["offset_std"],
                "max_offset": metrics["max_offset"],
                "min_offset": metrics["min_offset"],
                "offset_entropy": metrics["offset_entropy"],
                "zero_ratio": metrics["zero_ratio"],
                "mean_activation": feat_metrics["mean_activation"],
                "sparsity": feat_metrics["sparsity"],
                "energy": feat_metrics["energy"]
            })
            
        print(f"Processed {img_file}: predicted class {pred_class} (confidence: {conf_val:.4f})")
        
    # Write CSV summary
    csv_path = os.path.join(args.output, "scan_analysis_summary.csv")
    csv_keys = csv_rows[0].keys() if csv_rows else []
    with open(csv_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=csv_keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_rows)
        
    print(f"Analysis complete. Metrics summary saved to: {csv_path}")
    
    # Generate publication-ready aggregate plots
    if csv_rows:
        generate_aggregate_plots(csv_rows, args.output)

def generate_aggregate_plots(csv_rows, output_dir):
    """
    Generates publication-ready plots summarizing statistics across all analyzed images.
    """
    stages = sorted(list(set(r["stage"] for r in csv_rows)))
    
    # Plot 1: Mean Offset Magnitude per Stage for Correct vs. Incorrect Predictions
    correct_offsets = {s: [] for s in stages}
    incorrect_offsets = {s: [] for s in stages}
    
    for r in csv_rows:
        # Estimation of offset mean (proxy for magnitude)
        if r["correct"]:
            correct_offsets[r["stage"]].append(r["offset_mean"])
        else:
            incorrect_offsets[r["stage"]].append(r["offset_mean"])
            
    mean_correct = [np.mean(correct_offsets[s]) if correct_offsets[s] else 0.0 for s in stages]
    mean_incorrect = [np.mean(incorrect_offsets[s]) if incorrect_offsets[s] else 0.0 for s in stages]
    
    plt.figure(figsize=(8, 5))
    x = np.arange(len(stages))
    width = 0.35
    
    plt.bar(x - width/2, mean_correct, width, label='Correct Predictions', color='forestgreen', alpha=0.8)
    plt.bar(x + width/2, mean_incorrect, width, label='Incorrect Predictions', color='crimson', alpha=0.8)
    
    plt.xlabel('Backbone Stage')
    plt.ylabel('Mean Offset Magnitude')
    plt.title('Comparison of Scan Offsets: Correct vs. Incorrect Predictions')
    plt.xticks(x, [f"Stage {s}" for s in stages])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregate_correct_vs_incorrect_offsets.png"), dpi=200)
    plt.close()
    
    # Plot 2: Offset Entropy across Stages
    entropy_vals = {s: [] for s in stages}
    for r in csv_rows:
        entropy_vals[r["stage"]].append(r["offset_entropy"])
        
    mean_entropy = [np.mean(entropy_vals[s]) for s in stages]
    std_entropy = [np.std(entropy_vals[s]) for s in stages]
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(stages, mean_entropy, yerr=std_entropy, fmt='-o', color='purple', capsize=5, elinewidth=2, markeredgewidth=2)
    plt.xlabel('Backbone Stage')
    plt.ylabel('Offset Shannon Entropy')
    plt.title('DASSM Scan Offset Entropy across Stages')
    plt.xticks(stages, [f"Stage {s}" for s in stages])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregate_entropy_across_stages.png"), dpi=200)
    plt.close()
    
    print("Aggregate plots saved to output directory.")

if __name__ == "__main__":
    main()
