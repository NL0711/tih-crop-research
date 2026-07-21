import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
try:
    from .attention import SEBlock
except ImportError:
    from attention import SEBlock


class SEGradCAM:
    def __init__(self, model: nn.Module, targets: Optional[List[nn.Module]] = None):
        self.model = model
        self.hooks = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

        if targets is not None:
            self.targets = targets
        else:
            self.targets = self._auto_discover_targets()

        self._register_hooks()

    def _auto_discover_targets(self) -> List[nn.Module]:
        targets = []
        for i in range(1, 5):
            norm = getattr(self.model, f"norm{i}", None)
            if norm is not None:
                targets.append(norm)
        for name, module in self.model.named_modules():
            if isinstance(module, SEBlock):
                targets.append(module)
        return targets

    def _register_hooks(self):
        for module in self.targets:
            name = self._module_name(module)

            def make_forward_hook(n):
                def hook(mod, inp, out):
                    self.activations[n] = out
                return hook

            def make_backward_hook(n):
                def hook(mod, grad_inp, grad_out):
                    self.gradients[n] = grad_out[0]
                return hook

            fwd = module.register_forward_hook(make_forward_hook(name))
            bwd = module.register_full_backward_hook(make_backward_hook(name))
            self.hooks.extend([fwd, bwd])

    def _module_name(self, module: nn.Module) -> str:
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return str(id(module))

    def generate_cam(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        was_training = self.model.training
        self.model.eval()

        self.activations.clear()
        self.gradients.clear()

        with torch.enable_grad():
            input_var = input_tensor.clone().requires_grad_(True)
            output = self.model(input_var)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

        cams: Dict[str, np.ndarray] = {}
        for name in self.activations:
            if name not in self.gradients:
                continue
            activations = self.activations[name]
            gradients = self.gradients[name]
            if gradients is None:
                continue

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

            cams[name] = cam

        if not was_training:
            self.model.eval()
        else:
            self.model.train()

        return cams

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()

    def __del__(self):
        self.remove_hooks()


def _channel_mean(feat: torch.Tensor) -> np.ndarray:
    return feat.mean(dim=1, keepdim=True).squeeze().detach().cpu().numpy()


def _gradcam_overlay(cam: np.ndarray, image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * np.uint8(255 * image)).astype(np.uint8)


def compare_se_impact(
    model: nn.Module,
    image_tensor: torch.Tensor,
    stage_idx: int = 2,
    class_idx: Optional[int] = None,
    save_path: str = "se_impact.png",
):
    stage_names = {0: "norm1", 1: "norm2", 2: "norm3", 3: "norm4"}

    gradcam = SEGradCAM(model)
    cams = gradcam.generate_cam(image_tensor, class_idx)

    norm_name = stage_names.get(stage_idx)
    se_blocks_list = [(n, m) for n, m in model.named_modules() if isinstance(m, SEBlock)]

    if not se_blocks_list or stage_idx >= len(se_blocks_list) or norm_name is None:
        gradcam.remove_hooks()
        return

    se_name, se_block = se_blocks_list[stage_idx]
    has_after_gradcam = se_name in cams

    cam_before = cams.get(norm_name)
    cam_after = cams.get(se_name)

    feat_before = _channel_mean(se_block._input_feat)
    feat_after = _channel_mean(se_block._input_feat * se_block._channel_weights)
    feat_diff = feat_after - feat_before

    weights = se_block._channel_weights.squeeze().detach().cpu().numpy().flatten()
    weights_sorted = np.sort(weights)

    img_t = image_tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img_t - img_t.min()) / (img_t.max() - img_t.min())

    fig = plt.figure(figsize=(15, 14))

    ax = plt.subplot(3, 3, 1)
    im = ax.imshow(feat_before, cmap='viridis')
    ax.set_title(f"Feature Map Before SE (Stage {stage_idx})")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = plt.subplot(3, 3, 2)
    im = ax.imshow(feat_after, cmap='viridis')
    ax.set_title(f"Feature Map After SE (Stage {stage_idx})")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = plt.subplot(3, 3, 3)
    vmax = max(abs(feat_diff.min()), abs(feat_diff.max()))
    im = ax.imshow(feat_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title("Feature Map Difference")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    if cam_before is not None:
        ax = plt.subplot(3, 3, 4)
        overlay_before = _gradcam_overlay(cam_before, img)
        ax.imshow(overlay_before)
        ax.set_title(f"GradCAM Before SE (Stage {stage_idx})")
        ax.axis('off')
    else:
        ax = plt.subplot(3, 3, 4)
        ax.text(0.5, 0.5, 'GradCAM Before SE\nNot Available', ha='center', va='center')
        ax.set_title(f"GradCAM Before SE")
        ax.axis('off')

    if has_after_gradcam and cam_after is not None:
        ax = plt.subplot(3, 3, 5)
        overlay_after = _gradcam_overlay(cam_after, img)
        ax.imshow(overlay_after)
        ax.set_title(f"GradCAM After SE (Stage {stage_idx})")
        ax.axis('off')

        ax = plt.subplot(3, 3, 6)
        cam_diff = cam_after - cam_before
        vmax = max(abs(cam_diff.min()), abs(cam_diff.max()))
        im = ax.imshow(cam_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title("GradCAM Difference")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax = plt.subplot(3, 3, 5)
        colors = ['#e74c3c' if w > 1.0 else '#3498db' for w in weights_sorted]
        ax.bar(range(len(weights_sorted)), weights_sorted, color=colors, width=1.0)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel("Channel Index (sorted)")
        ax.set_ylabel("Weight")
        ax.set_title(f"SE Channel Weights ({len(weights)} ch)")
        ax.set_xlim(0, len(weights_sorted))

        ax = plt.subplot(3, 3, 6)
        ax.text(0.5, 0.5, 'GradCAM After SE\nNot Available\n(no gradient flow\nthrough this stage)', 
                ha='center', va='center', fontsize=9)
        ax.set_title("GradCAM After SE")
        ax.axis('off')

    ax = plt.subplot(3, 1, 3)
    colors = ['#e74c3c' if w > 1.0 else '#3498db' for w in weights_sorted]
    ax.bar(range(len(weights_sorted)), weights_sorted, color=colors, width=1.0)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='1.0 (identity)')
    ax.set_xlabel("Channel Index (sorted)")
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"SE Channel Weights — Stage {stage_idx} ({len(weights)} channels)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(weights_sorted))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    gradcam.remove_hooks()


def compare_all_stages(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    save_path: str = "se_all_stages.png",
):
    gradcam = SEGradCAM(model)
    cams = gradcam.generate_cam(image_tensor, class_idx)

    se_blocks = []
    for name, module in model.named_modules():
        if isinstance(module, SEBlock):
            se_blocks.append((name, module))

    num_stages = len(se_blocks)
    img_t = image_tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img_t - img_t.min()) / (img_t.max() - img_t.min())

    num_cols = max(num_stages * 2, 4)
    fig = plt.figure(figsize=(3 * num_cols, 2 * num_stages + 4))

    hist_rows = num_stages
    grad_rows = 2
    total_rows = 1 + hist_rows + grad_rows

    ax = plt.subplot2grid((total_rows, num_cols), (0, 0), colspan=num_cols)
    ax.imshow(img)
    ax.set_title("Input Image")
    ax.axis('off')

    for i, (se_name, se_block) in enumerate(se_blocks):
        weights = se_block._channel_weights.squeeze().detach().cpu().numpy().flatten()
        weights_sorted = np.sort(weights)

        ax = plt.subplot2grid((total_rows, num_cols), (i + 1, 0), colspan=num_cols)
        colors = ['#e74c3c' if w > 1.0 else '#3498db' for w in weights_sorted]
        ax.bar(range(len(weights_sorted)), weights_sorted, color=colors, width=1.0)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f"S{i} ({se_block.channels}ch)")
        ax.set_ylim(0, max(weights_sorted.max(), 1.1))
        ax.set_xlim(0, len(weights_sorted))
        ax.tick_params(axis='x', labelsize=6)
        ax.set_xlabel("Channel Index (sorted)", fontsize=7)

    norm_names = [f"norm{i+1}" for i in range(num_stages)]
    grad_row_start = 1 + hist_rows
    for i, (norm_name, (se_name, se_block)) in enumerate(zip(norm_names, se_blocks)):
        cam_before = cams.get(norm_name)

        col_before = i * 2
        col_after = i * 2 + 1

        ax = plt.subplot2grid((total_rows, num_cols), (grad_row_start, col_before))
        if cam_before is not None:
            overlay_before = _gradcam_overlay(cam_before, img)
            ax.imshow(overlay_before)
            ax.set_title(f"Stage {i} Before SE", fontsize=8)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            ax.set_title(f"Stage {i} Before SE", fontsize=8)
        ax.axis('off')

        ax = plt.subplot2grid((total_rows, num_cols), (grad_row_start, col_after))
        cam_after = cams.get(se_name)
        if cam_after is not None:
            overlay_after = _gradcam_overlay(cam_after, img)
            ax.imshow(overlay_after)
            ax.set_title(f"Stage {i} After SE", fontsize=8)
        else:
            feat_viz = _channel_mean(se_block._input_feat * se_block._channel_weights)
            im = ax.imshow(feat_viz, cmap='viridis')
            ax.set_title(f"Stage {i} SE Out Feat", fontsize=8)
        ax.axis('off')

    plt.suptitle("SE Impact Across All Stages", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    gradcam.remove_hooks()
