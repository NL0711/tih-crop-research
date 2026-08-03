import torch
import torch.nn as nn
from typing import List


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block that adaptively recalibrates channel-wise feature responses."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bottleneck MLP: C -> C/reduction -> C
        hidden_dim = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        self._input_feat = x
        
        # Squeeze: Global average pooling
        squeezed = self.global_avg_pool(x)  # (B, C, 1, 1)
        squeezed = squeezed.view(B, C)  # (B, C)
        
        # Excitation: MLP to get channel attention weights
        channel_weights = self.mlp(squeezed)  # (B, C)
        channel_weights = channel_weights.view(B, C, 1, 1)  # (B, C, 1, 1)
        self._channel_weights = channel_weights
        
        # Scale: Multiply weights back onto input
        return x * channel_weights


class CBAM(nn.Module):
    """Convolutional Block Attention Module that applies channel and spatial attention sequentially."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Channel attention components
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_dim = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()
        
        # Spatial attention components
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Channel attention
        avg_pooled = self.global_avg_pool(x).view(B, C)  # (B, C)
        max_pooled = self.global_max_pool(x).view(B, C)  # (B, C)
        
        # Share MLP weights for both pooling paths
        avg_out = self.channel_mlp(avg_pooled)  # (B, C)
        max_out = self.channel_mlp(max_pooled)  # (B, C)
        
        # Sum and apply sigmoid
        channel_attention = self.channel_sigmoid(avg_out + max_out)  # (B, C)
        channel_attention = channel_attention.view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Apply channel attention
        x = x * channel_attention  # (B, C, H, W)
        
        # Spatial attention
        # Channel-wise average and max pooling across channel dimension
        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and apply 7x7 conv
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, H, W)
        spatial_attention = self.spatial_sigmoid(self.spatial_conv(spatial_input))  # (B, 1, H, W)
        
        # Apply spatial attention
        x = x * spatial_attention  # (B, C, H, W)
        
        return x


class StageAttentionWrapper(nn.Module):
    """Wrapper that applies attention modules (SEBlock or CBAM) to multi-stage feature maps."""
    
    def __init__(self, stage_channels: List[int], use_se_only: bool = False, reduction: int = 16):
        super().__init__()
        self.stage_channels = stage_channels
        self.use_se_only = use_se_only
        self.reduction = reduction
        
        # Create attention module for each stage
        AttentionModule = SEBlock if use_se_only else CBAM
        self.attention_modules = nn.ModuleList([
            AttentionModule(channels, reduction=reduction)
            for channels in stage_channels
        ])
    
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply attention to each stage's feature map.
        
        Args:
            feats: List of feature maps, one per stage
            
        Returns:
            List of attention-enhanced feature maps, same order and shapes as input
        """
        if len(feats) != len(self.attention_modules):
            raise ValueError(
                f"Number of feature maps ({len(feats)}) does not match "
                f"number of attention modules ({len(self.attention_modules)})"
            )
        
        enhanced_feats = []
        for feat, attention_module in zip(feats, self.attention_modules):
            enhanced = attention_module(feat)
            enhanced_feats.append(enhanced)
        
        return enhanced_feats


if __name__ == "__main__":
    # Test StageAttentionWrapper with typical DAMamba stage dimensions
    stage_channels = [96, 192, 384, 768]
    
    # Test with CBAM (default)
    print("Testing StageAttentionWrapper with CBAM:")
    wrapper_cbam = StageAttentionWrapper(stage_channels, use_se_only=False)
    
    # Create dummy tensors with shapes matching DAMamba stages
    dummy_feats = [
        torch.randn(2, 96, 56, 56),
        torch.randn(2, 192, 28, 28),
        torch.randn(2, 384, 14, 14),
        torch.randn(2, 768, 7, 7)
    ]
    
    print("Input shapes:")
    for i, feat in enumerate(dummy_feats):
        print(f"  Stage {i}: {feat.shape}")
    
    enhanced_feats_cbam = wrapper_cbam(dummy_feats)
    
    print("Output shapes (CBAM):")
    for i, feat in enumerate(enhanced_feats_cbam):
        print(f"  Stage {i}: {feat.shape}")
    
    # Verify shapes match
    for i, (inp, out) in enumerate(zip(dummy_feats, enhanced_feats_cbam)):
        assert inp.shape == out.shape, f"Shape mismatch at stage {i}: {inp.shape} vs {out.shape}"
    print("✓ All output shapes match input shapes (CBAM)")
    
    # Test with SEBlock
    print("\nTesting StageAttentionWrapper with SEBlock:")
    wrapper_se = StageAttentionWrapper(stage_channels, use_se_only=True)
    enhanced_feats_se = wrapper_se(dummy_feats)
    
    print("Output shapes (SEBlock):")
    for i, feat in enumerate(enhanced_feats_se):
        print(f"  Stage {i}: {feat.shape}")
    
    # Verify shapes match
    for i, (inp, out) in enumerate(zip(dummy_feats, enhanced_feats_se)):
        assert inp.shape == out.shape, f"Shape mismatch at stage {i}: {inp.shape} vs {out.shape}"
    print("✓ All output shapes match input shapes (SEBlock)")
    
    print("\nAll tests passed!")
