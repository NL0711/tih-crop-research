import os
import numpy as np
import matplotlib.pyplot as plt
import torch

class DASVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, image_np, image_name, debug_data):
        """
        Generates and saves the 5 required png visual plots for each stage/block.
        
        image_np: input image as numpy array of shape (H, W, 3) or (3, H, W)
        image_name: string identifier
        debug_data: list of dictionaries recorded during inference
        """
        # Ensure image is (H, W, 3) and in [0, 255] or [0, 1]
        if image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
            
        # Normalize image for plotting
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
            
        base_out = os.path.join(self.output_dir, image_name)
        os.makedirs(base_out, exist_ok=True)
        
        for record in debug_data:
            stage_idx = record["stage"]
            block_idx = record["block"]
            feat_h, feat_w = record["H"], record["W"]
            
            stage_dir = os.path.join(base_out, f"stage_{stage_idx}_block_{block_idx}")
            os.makedirs(stage_dir, exist_ok=True)
            
            # Extract records
            raw_offset = record["raw_offset"].numpy()      # (H, W, group, num_points, 2)
            base_grid = record["base_grid"].numpy()        # (H, W, group, num_points, 2)
            sampling_grid = record["sampling_grid"].numpy() # (H, W, group, num_points, 2)
            
            # Heatmap of offset magnitudes
            offset_mag = record["offset_magnitude"].numpy() # (H, W, group, num_points)
            # Average over groups and points to get a single spatial heatmap
            spatial_heatmap = np.mean(offset_mag, axis=(2, 3)) # (H, W)
            
            # --- 1. offset_heatmap.png ---
            plt.figure(figsize=(6, 5))
            plt.imshow(spatial_heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Offset Magnitude')
            plt.title(f"Stage {stage_idx} Block {block_idx} Offset Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "offset_heatmap.png"), dpi=150)
            plt.close()
            
            # --- 2. scan_vectors.png ---
            # Plot offsets as quiver vectors (average offset direction per pixel)
            # average coordinates along group and num_points
            avg_offset_y = np.mean(raw_offset[..., 1], axis=(2, 3)) # (H, W)
            avg_offset_x = np.mean(raw_offset[..., 0], axis=(2, 3)) # (H, W)
            
            X, Y = np.meshgrid(np.arange(feat_w), np.arange(feat_h))
            
            plt.figure(figsize=(8, 8))
            # Resize image to match feature resolution for background overlay
            plt.imshow(image_np, extent=(0, feat_w - 1, feat_h - 1, 0), alpha=0.6)
            plt.quiver(X, Y, avg_offset_x, avg_offset_y, color='cyan', scale=50, width=0.005)
            plt.title(f"Stage {stage_idx} Block {block_idx} Scan Vectors")
            plt.xlim(-0.5, feat_w - 0.5)
            plt.ylim(feat_h - 0.5, -0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "scan_vectors.png"), dpi=150)
            plt.close()
            
            # --- 3. sampling_grid.png ---
            # Show a grid plot of base vs sampled points for a subset of points (e.g. center area to avoid clutter)
            plt.figure(figsize=(8, 8))
            ch, cw = feat_h // 2, feat_w // 2
            # Crop a 3x3 patch in the center of the feature map to display grid points clearly
            h_slice = slice(max(0, ch - 1), min(feat_h, ch + 2))
            w_slice = slice(max(0, cw - 1), min(feat_w, cw + 2))
            
            # Flattens points for base and sampling grids
            # Shape is (H_sub, W_sub, group, num_points, 2)
            sub_base = base_grid[h_slice, w_slice].reshape(-1, 2)
            sub_samp = sampling_grid[h_slice, w_slice].reshape(-1, 2)
            
            plt.scatter(sub_base[:, 0], sub_base[:, 1], color='red', alpha=0.6, label='Base Grid', s=40)
            plt.scatter(sub_samp[:, 0], sub_samp[:, 1], color='blue', alpha=0.6, label='Sampled Grid', s=40)
            
            # Draw lines connecting base to sampled grid points
            for i in range(len(sub_base)):
                plt.plot([sub_base[i, 0], sub_samp[i, 0]], [sub_base[i, 1], sub_samp[i, 1]], 'gray', linestyle='--', alpha=0.5)
                
            plt.title(f"Stage {stage_idx} Block {block_idx} Base vs Sampled Grid (Center Crop)")
            plt.legend()
            plt.grid(True)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "sampling_grid.png"), dpi=150)
            plt.close()
            
            # --- 4. feature_before_after.png ---
            # Side-by-side mean channel activation of input vs output
            in_feat = record["input"].numpy() # (C, H, W)
            out_feat = record["scanned"].numpy() # (H, W, C) or permuted. Wait! scanned is (H, W, C).
            # Let's make sure shape matches
            if in_feat.ndim == 3:
                in_feat_mean = np.mean(in_feat, axis=0) # (H, W)
            else:
                in_feat_mean = in_feat
                
            if out_feat.ndim == 3:
                # if channel is last
                if out_feat.shape[0] == feat_h:
                    out_feat_mean = np.mean(out_feat, axis=-1)
                else:
                    out_feat_mean = np.mean(out_feat, axis=0)
            else:
                out_feat_mean = out_feat
                
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            im1 = axes[0].imshow(in_feat_mean, cmap='viridis')
            axes[0].set_title("Input Feature (Mean Activation)")
            fig.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(out_feat_mean, cmap='viridis')
            axes[1].set_title("Scanned Feature (Mean Activation)")
            fig.colorbar(im2, ax=axes[1])
            
            plt.suptitle(f"Stage {stage_idx} Block {block_idx} Feature Transformation")
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "feature_before_after.png"), dpi=150)
            plt.close()
            
            # --- 5. offset_histogram.png ---
            # Histogram of offset magnitudes and dx, dy coordinate distributions
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.hist(offset_mag.flatten(), bins=30, color='purple', alpha=0.7, edgecolor='black')
            plt.title("Offset Magnitude Distribution")
            plt.xlabel("Magnitude")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 2, 2)
            plt.hist(raw_offset[..., 0].flatten(), bins=30, color='blue', alpha=0.5, label='dx', edgecolor='black')
            plt.hist(raw_offset[..., 1].flatten(), bins=30, color='green', alpha=0.5, label='dy', edgecolor='black')
            plt.title("dx and dy Coordinate Distributions")
            plt.xlabel("Offset Value")
            plt.ylabel("Frequency")
            plt.legend()
            
            plt.suptitle(f"Stage {stage_idx} Block {block_idx} Offset Histograms")
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "offset_histogram.png"), dpi=150)
            plt.close()
