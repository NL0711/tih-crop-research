import os
import json
import numpy as np

def export_debug_data(output_dir, image_name, debug_data, metadata_extra):
    """
    Saves recorded debug coordinates, feature statistics, and metadata to disk.
    
    debug_data is a list of dictionary records collected from Dynamic_Adaptive_Scan forward passes.
    metadata_extra contains run-specific metadata:
      - predicted_class
      - confidence
      - true_label
      - correct
      - input_image_size (tuple H, W)
    """
    base_out = os.path.join(output_dir, image_name)
    os.makedirs(base_out, exist_ok=True)
    
    # Save a master metadata structure
    master_metadata = {
        "image_name": image_name,
        "predicted_class": metadata_extra.get("predicted_class"),
        "confidence": float(metadata_extra.get("confidence", 0.0)),
        "true_label": metadata_extra.get("true_label"),
        "correct": bool(metadata_extra.get("correct", False)),
        "input_image_size": metadata_extra.get("input_image_size"),
        "stages": []
    }
    
    for idx, record in enumerate(debug_data):
        stage_idx = record["stage"]
        block_idx = record["block"]
        feat_h = record["H"]
        feat_w = record["W"]
        
        stage_dir = os.path.join(base_out, f"stage_{stage_idx}_block_{block_idx}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Extract features and grids
        raw_offset = record["raw_offset"].numpy()  # (H, W, group, num_points, 2)
        base_grid = record["base_grid"].numpy()    # (H, W, group, num_points, 2)
        sampling_grid = record["sampling_grid"].numpy() # (H, W, group, num_points, 2)
        
        # Save main arrays
        np.save(os.path.join(stage_dir, "offset.npy"), raw_offset)
        np.save(os.path.join(stage_dir, "grid.npy"), sampling_grid)
        np.save(os.path.join(stage_dir, "base_grid.npy"), base_grid)
        
        # Save group-wise arrays
        num_groups = record["group"]
        for g in range(num_groups):
            group_offset = raw_offset[:, :, g, :, :]
            np.save(os.path.join(stage_dir, f"group_{g}.npy"), group_offset)
            
        # Compile stage-specific metadata
        img_h, img_w = master_metadata["input_image_size"]
        downsample_ratio = float(img_h / feat_h)
        
        stage_meta = {
            "stage": stage_idx,
            "block": block_idx,
            "feature_size": [feat_h, feat_w],
            "downsample_ratio": downsample_ratio,
            "kernel_size": record["kernel_size"],
            "group": num_groups,
            "offset_scale": float(record["offset_scale"]),
            "offset_metrics": record["offset_metrics"],
            "feature_metrics": record["feature_metrics"]
        }
        
        # Save stage-specific json
        with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
            json.dump(stage_meta, f, indent=4)
            
        master_metadata["stages"].append({
            "stage": stage_idx,
            "block": block_idx,
            "dir": f"stage_{stage_idx}_block_{block_idx}",
            "downsample_ratio": downsample_ratio,
            "offset_metrics": record["offset_metrics"],
            "feature_metrics": record["feature_metrics"]
        })
        
    # Write master metadata
    with open(os.path.join(base_out, "metadata.json"), "w") as f:
        json.dump(master_metadata, f, indent=4)
