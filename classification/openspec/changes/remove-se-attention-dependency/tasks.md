## 1. Remove attention from DAMamba architecture

- [x] 1.1 Remove the `StageAttentionWrapper` import block from `models/DAMamba.py`
- [x] 1.2 Remove `use_attention` and `use_se_only` parameters from `DAMamba.__init__` signature
- [x] 1.3 Remove `self.use_attention` assignment and the `if self.use_attention:` / `else: self.attention = None` branch
- [x] 1.4 Rewrite `forward_features()` to return the final normalized stage feature directly (stem → stages → norms), removing the `outs` list and the `self.attention(outs)` call
- [x] 1.5 Verify no dead attention branches remain in the main architecture forward path

## 2. Remove SE/attention model files

- [x] 2.1 Delete `models/attention.py` (`SEBlock`, `CBAM`, `StageAttentionWrapper`)
- [x] 2.2 Delete `models/gradcam.py` (`SEGradCAM`, `compare_se_impact`, `compare_all_stages`)
- [x] 2.3 Delete `utils/generate_pseudo_lesions.py` (SEGradCAM-based pseudo-lesion generator)

## 3. Update model construction

- [x] 3.1 Remove `use_attention=config.MODEL.DAMAMBA.USE_ATTENTION` and `use_se_only=config.MODEL.DAMAMBA.USE_SE_ONLY` from `build_model()` in `models/__init__.py`
- [x] 3.2 Remove `use_attention=False` from `tests/test_debug_pipeline.py` so the test matches the new signature

## 4. Remove obsolete configuration

- [x] 4.1 Remove `_C.MODEL.DAMAMBA.USE_ATTENTION` and `_C.MODEL.DAMAMBA.USE_SE_ONLY` from `config.py`
- [x] 4.2 Remove `USE_ATTENTION:` and `USE_SE_ONLY:` from `configs/DAMamba/damamba_tiny.yaml`
- [x] 4.3 Remove `USE_ATTENTION:` and `USE_SE_ONLY:` from `configs/DAMamba/damamba_small.yaml`
- [x] 4.4 Remove `USE_ATTENTION:` and `USE_SE_ONLY:` from `configs/DAMamba/damamba_base.yaml`

## 5. Verify the result

- [x] 5.1 Run syntax/import checks (e.g. `python -c "import models.DAMamba"` without `attention` module present)
- [x] 5.2 Instantiate `DAMamba_T` and run a dummy forward pass; verify logits shape `[B, num_classes]`
- [x] 5.3 Instantiate `DAMamba_S` and `DAMamba_B` if available; verify they construct and forward
- [x] 5.4 Confirm no `StageAttentionWrapper` executes and no `SEGradCAM`/`SEBlock` is imported by the model
- [x] 5.5 Confirm `Dynamic_Adaptive_Scan` still executes (e.g. run `tests/test_debug_pipeline.py` or inspect debug records)
- [x] 5.6 Run the existing test suite (`pytest tests/`) and confirm green
- [x] 5.7 Load a baseline vanilla checkpoint as far as possible and report missing/unexpected keys
- [x] 5.8 Grep the repo for remaining SE/attention references and report them (including `config.py` LADAS block that stays intact)

## 6. Final state documentation

- [x] 6.1 Report files changed, files removed/deprecated, config changes, checkpoint compatibility notes, and verification results