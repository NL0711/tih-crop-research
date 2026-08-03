import torch
import torch.nn as nn
import torch.nn.functional as F

def get_approximate_grid(H, W, stride, pad, kernel_size, dilation, group, num_points, device, dtype):
    """
    Computes a simplified approximate base grid for visualization.
    """
    # Grid of relative offsets (dy, dx)
    kh, kw = kernel_size, kernel_size
    dy = torch.arange(-(kh - 1) // 2, (kh - 1) // 2 + 1, device=device, dtype=dtype)
    dx = torch.arange(-(kw - 1) // 2, (kw - 1) // 2 + 1, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (K^2, 2) - use (x, y) order
    
    # Output pixel centers mapping to input
    ys = torch.arange(H, device=device, dtype=dtype) * stride - pad
    xs = torch.arange(W, device=device, dtype=dtype) * stride - pad
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
    pixel_centers = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2) - (x, y) order
    
    # Base grid: (H, W, group, num_points, 2)
    base_grid = pixel_centers[:, :, None, None, :] + grid[None, None, None, :num_points, :] * dilation
    base_grid = base_grid.expand(-1, -1, group, -1, -1)
    return base_grid

def _get_reference_points(spatial_shapes, device, dtype, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 - pad_h,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 - pad_h + (H_out - 1) * stride_h,
            H_out,
            dtype=dtype,
            device=device),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 - pad_w,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 - pad_w + (W_out - 1) * stride_w,
            W_out,
            dtype=dtype,
            device=device),
        indexing='ij'
    )
    
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)
    return ref

def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, remove_center, device, dtype):
    _, H_, W_, _ = spatial_shapes
    
    dy = torch.arange(-(kernel_h - 1) // 2, (kernel_h - 1) // 2 + 1, device=device, dtype=dtype)
    dx = torch.arange(-(kernel_w - 1) // 2, (kernel_w - 1) // 2 + 1, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
    grid = torch.stack([grid_x, grid_y], -1).view(-1, 2) # DCNv3 uses (x, y) order for grids/offsets internally in C++
    
    if remove_center:
        center_idx = (kernel_h * kernel_w) // 2
        grid = torch.cat([grid[:center_idx], grid[center_idx+1:]], dim=0)
        
    grid = grid.reshape(1, 1, 1, -1, 2) * torch.tensor([dilation_w, dilation_h], device=device, dtype=dtype).view(1, 1, 1, 1, 2)
    
    # normalize by H_ and W_
    grid[..., 0] = grid[..., 0] / W_
    grid[..., 1] = grid[..., 1] / H_
    return grid

def get_dcnv3_grid(spatial_shapes, offset, kernel_size, stride, pad, dilation, group, offset_scale, remove_center, device, dtype):
    """
    Computes the exact coordinate sampling grid based on DCNv3 reference points and dilation offsets.
    Returns normalized coordinates (in range [0, 1]) and absolute coordinates.
    """
    kh, kw = kernel_size, kernel_size
    ref = _get_reference_points(
        spatial_shapes, device, dtype, kh, kw, dilation, dilation, pad, pad, stride, stride
    ) # (1, H_out, W_out, 1, 2)
    
    dilation_grid = _generate_dilation_grids(
        spatial_shapes, kh, kw, dilation, dilation, group, remove_center, device, dtype
    ) # (1, 1, 1, num_points, 2)
    
    # Base grid in DCNv3 (normalized coordinates)
    base_grid_normalized = ref + dilation_grid # Broadcasts to (1, H_out, W_out, num_points, 2)
    base_grid_normalized = base_grid_normalized.unsqueeze(3).expand(-1, -1, -1, group, -1, -1) # (1, H_out, W_out, group, num_points, 2)
    
    # Reshape offset to (1, H_out, W_out, group, num_points, 2)
    # Internally offset has shape (N, H_out, W_out, group * num_points * 2)
    num_points = kh * kw - remove_center
    offset_reshaped = offset.view(1, spatial_shapes[1], spatial_shapes[2], group, num_points, 2)
    
    # Adjust for normalization division in DCNv3 grid calculation
    # offsets in DCNv3 are normalized by spatial_shapes (W_, H_) when added to ref
    offset_normalized = offset_reshaped.clone()
    _, H_, W_, _ = spatial_shapes
    offset_normalized[..., 0] = offset_normalized[..., 0] / W_
    offset_normalized[..., 1] = offset_normalized[..., 1] / H_
    
    # Sampling grid (normalized)
    sampling_grid_normalized = base_grid_normalized + offset_normalized * offset_scale
    
    # Scale to absolute coordinates
    scale_tensor = torch.tensor([W_, H_], device=device, dtype=dtype).view(1, 1, 1, 1, 1, 2)
    base_grid_abs = base_grid_normalized * scale_tensor
    sampling_grid_abs = sampling_grid_normalized * scale_tensor
    
    return {
        "base_grid_normalized": base_grid_normalized[0],
        "sampling_grid_normalized": sampling_grid_normalized[0],
        "base_grid_abs": base_grid_abs[0],
        "sampling_grid_abs": sampling_grid_abs[0]
    }
