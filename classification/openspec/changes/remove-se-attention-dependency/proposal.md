## Why

The DAMamba architecture and LADAS pipeline currently carry an SE/attention dependency (`StageAttentionWrapper`, `SEBlock`/`CBAM`, `SEGradCAM`) that was used for disease-guidance. This dependency must be stripped out before the new LADAS spatial-selective guidance mechanism is implemented, so the repository represents a clean "Vanilla DAMamba + Dynamic Adaptive Scan + DASSM + classifier" baseline with no SE/attention requirement.

## What Changes

- **BREAKING** Remove `StageAttentionWrapper`, `SEBlock`, `CBAM`, and attention-based feature processing from the main DAMamba architecture.
- Remove `self.use_attention` / `self.use_se_only` flags and the `if self.use_attention: self.attention = ... else: self.attention = None` initialization branch from `DAMamba`.
- Simplify `forward_features()` so it returns the final normalized stage feature directly, without passing the feature list through `StageAttentionWrapper`.
- **BREAKING** Remove the `SEGradCAM` dependency from LADAS; remove SEGradCAM-based pseudo-lesion generation from `utils/generate_pseudo_lesions.py`.
- Remove now-obsolete configuration options (`MODEL.DAMAMBA.USE_ATTENTION`, `MODEL.DAMAMBA.USE_SE_ONLY`) from `config.py` defaults and all YAML configs.
- Remove attention assumptions from model construction in `models/__init__.py` and from `tests/test_debug_pipeline.py`.
- Preserve `Dynamic_Adaptive_Scan`, `DASSM`, `DCNv3`, offset generation, stage normalization, staging, and the `MlpHead` classifier unchanged.
- Do **not** implement the replacement LADAS spatial-selective guidance mechanism in this change.

## Capabilities

### New Capabilities
- `damamba-core-architecture`: Defines the vanilla DAMamba forward flow (stem → stages → norms → classifier) with no SE/attention requirement, preserving Dynamic Adaptive Scan and stage normalization behavior.

### Modified Capabilities
<!-- No existing spec is changed. The existing specs (`scan-viz-grid-fidelity`, `scan-viz-paper-figure`) are orthogonal to SE/attention and are unaffected. -->

## Impact

- **Code**: `models/DAMamba.py`, `models/attention.py`, `models/gradcam.py`, `models/__init__.py`, `utils/generate_pseudo_lesions.py`, `config.py`, `tests/test_debug_pipeline.py`.
- **Config**: `configs/DAMamba/damamba_tiny.yaml`, `damamba_small.yaml`, `damamba_base.yaml`.
- **Checkpoints**: Checkpoints containing `attention.*` keys become partially loadable; unused keys must be reported, not silently modified.
- **Pseudo-lesion pipeline**: SEGradCAM-based generator is deprecated/isolated; downstream `LADASDatasetWrapper` (data/build.py) and `get_lesion_logits()`/aux-loss plumbing are untouched and remain compatible.
- **Training flow**: image → DAMamba → classification logits → classification loss, with no SE/attention dependency.