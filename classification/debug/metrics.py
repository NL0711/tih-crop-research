import torch
import numpy as np

def compute_offset_metrics(offset_tensor):
    """
    Computes statistical metrics for offset coordinates on CPU.
    Input offset_tensor has shape: (H, W, group, num_points, 2)
    """
    offset_np = offset_tensor.float().numpy()
    
    # 2D coordinates
    dx = offset_np[..., 0]
    dy = offset_np[..., 1]
    
    # Magnitude: L2 norm of coordinates
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Basic Stats
    offset_mean = float(np.mean(offset_np))
    offset_std = float(np.std(offset_np))
    max_offset = float(np.max(offset_np))
    min_offset = float(np.min(offset_np))
    
    # Sparsity/Zero ratio
    zero_ratio = float(np.mean(np.abs(offset_np) < 1e-4))
    
    # Angle calculation: theta = atan2(dy, dx)
    theta = np.arctan2(dy, dx) # value in [-pi, pi]
    
    # Direction Histogram (e.g. 8 bins)
    hist, bin_edges = np.histogram(theta, bins=8, range=(-np.pi, np.pi))
    direction_histogram = hist.tolist()
    
    # Entropy of offset magnitudes
    # First normalize magnitudes to a probability distribution (hist)
    mag_hist, _ = np.histogram(magnitude, bins=10, range=(0.0, max(1.0, float(np.max(magnitude)))))
    mag_prob = mag_hist / (np.sum(mag_hist) + 1e-9)
    entropy = float(-np.sum(mag_prob * np.log2(mag_prob + 1e-9)))
    
    return {
        "offset_mean": offset_mean,
        "offset_std": offset_std,
        "max_offset": max_offset,
        "min_offset": min_offset,
        "zero_ratio": zero_ratio,
        "offset_entropy": entropy,
        "direction_histogram": direction_histogram,
        "bin_edges": bin_edges.tolist()
    }

def compute_feature_metrics(feature_tensor):
    """
    Computes statistics on feature activations.
    Input feature_tensor has shape: (C, H, W)
    """
    feat_np = feature_tensor.float().numpy()
    
    mean_act = float(np.mean(feat_np))
    std_act = float(np.std(feat_np))
    sparsity = float(np.mean(np.abs(feat_np) < 1e-5))
    energy = float(np.linalg.norm(feat_np)) # L2 norm/energy
    
    return {
        "mean_activation": mean_act,
        "std_activation": std_act,
        "sparsity": sparsity,
        "energy": energy
    }
