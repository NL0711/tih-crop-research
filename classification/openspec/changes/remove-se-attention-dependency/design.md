## Context

The DAMamba classification model (`models/DAMamba.py`) currently carries an optional SE/attention path:
`DAMamba.__init__` accepts `use_attention` / `use_se_only`, and when enabled registers a
`StageAttentionWrapper` (`models/attention.py`) that applies `SEBlock` / `CBAM` to the per-stage
feature maps before the classifier. `forward_features()` collects all normalized stage outputs
(`outs`) and, when attention is active, routes them through the wrapper.

A parallel disease-guidance pipeline (`SEGradCAM` in `models/gradcam.py`, and
`utils/generate_pseudo_lesions.py`) depends on `SEBlock` modules being present inside the model to
produce pseudo-lesion maps for LADAS training.

The dynamic scanning machinery (`Dynamic_Adaptive_Scan`, `DASSM`, `DCNv3`, offset prediction) is
independent of SE/attention and must stay functionally identical.

## Goals / Non-Goals

**Goals:**
- Remove all SE/attention dependency from the main DAMamba forward path (architecture + config + construction).
- Remove `SEGradCAM` and the SEGradCAM-based pseudo-lesion generator from the LADAS pipeline.
- Leave the repository in a clean state: Vanilla DAMamba + Dynamic Adaptive Scan + DASSM/SSM + classifier.
- Preserve vanilla DAMamba checkpoint compatibility and report (not silently alter) orphaned keys.

**Non-Goals:**
- Do NOT implement the new LADAS spatial-selective guidance mechanism.
- Do NOT modify `Dynamic_Adaptive_Scan`, `DASSM`, `DCNv3`, offset generation, or scan-debug behavior.
- Do NOT change LADAS core plumbing (`use_ladas`, `lesion_conv`, `get_lesion_logits()`,
  `LADASDatasetWrapper`, aux-loss in `main.py`) — these are downstream consumers of the (removed)
  generator but are not themselves SE/attention components and remain compatible.

## Decisions

### D1. Remove attention from `DAMamba` (`models/DAMamba.py`)
- Delete the `use_attention` and `use_se_only` constructor parameters.
- Delete `self.use_attention`, the `if self.use_attention: self.attention = StageAttentionWrapper(...) else: self.attention = None` branch.
- Delete the `StageAttentionWrapper` import block (lines ~23-26).
- Rewrite `forward_features()` to run `stem → stage → norm` for each stage and return the final
  normalized feature directly, with no `outs` list and no wrapper call.
- **Alternative considered**: keeping the disabled branch (`attention=None`) for future re-use. Rejected
  because it leaves dead code and a config surface that contradicts the desired "no SE/attention" state.

### D2. Remove `models/attention.py` entirely
- `SEBlock`, `CBAM`, and `StageAttentionWrapper` exist only to service the removed attention path and
  `SEGradCAM`. No non-SE component references them. Delete the file.
- **Alternative considered**: keeping it as an isolated/deprecated module. Rejected: nothing injects it
  into the model anymore, so it could not be loaded into `DAMamba`; leaving it would create a misleading
  "still-used" impression.

### D3. Remove `SEGradCAM` and SE-specific tooling from `models/gradcam.py`
- `SEGradCAM` (`_auto_discover_targets` scans for `SEBlock`), `compare_se_impact`, and
  `compare_all_stages` are exclusively SE-focused. Delete the file.
- **Alternative considered**: retaining generic `load_model` / `load_image` helpers. Rejected: they are
  only consumed by the SE-GradCAM CLI and the removed generator; no unrelated utility depends on them.
  If needed later, the generic checkpoint loader can be reintroduced at that point.

### D4. Remove the SEGradCAM-based pseudo-lesion generator (`utils/generate_pseudo_lesions.py`)
- The script is exclusively LADAS/SE-specific (its stated purpose is generating SEGradCAM-based maps).
  Delete it. Do not ship a stub claiming to generate maps without SEGradCAM.
- The LADAS data wrapper (`LADASDatasetWrapper`) reads existing `.npy` files from `PSEUDO_DIR`; it does
  not import SEGradCAM. It remains untouched and degrades gracefully (zero maps) when no `.npy` exist.

### D5. Remove obsolete configuration
- Delete `_C.MODEL.DAMAMBA.USE_ATTENTION` and `_C.MODEL.DAMAMBA.USE_SE_ONLY` from `config.py` defaults.
- Delete `USE_ATTENTION:` / `USE_SE_ONLY:` from `configs/DAMamba/*.yaml` (tiny, small, base).
- Remove `use_attention=...` / `use_se_only=...` from `build_model()` in `models/__init__.py`.
- Remove `use_attention=False` from `tests/test_debug_pipeline.py` so the test matches the new signature.
- Retain all other config (LADAS block, DDP, TRAIN, etc.) unchanged.

### D6. Preserve architectural invariants
- Stage index / block index assignment to `Dynamic_Adaptive_Scan` modules must remain.
- Scan-debug (`enable_scan_debug`, `disable_scan_debug`, `get_scan_debug_data`), `get_lesion_logits`,
  `set_grad_checkpointing`, `no_weight_decay` remain unchanged.
- `norm1..norm4` modules are preserved; final classifier input equals `norm4` output.

### D7. Configuration precedence for the config merge
- The `DAMamba.STAGE` name refers to a single model `DAMamba`, with the `use_ladas`/`use_offset_residual` fields already separate.

### D8. Checkpoint compatibility
- `DAMamba` was trained with `use_attention=False` by default → vanilla checkpoints contain no
  `attention.*` keys and load cleanly.
- Checkpoints produced with attention enabled carry `attention.attention_modules.*.mlp.*` keys that
  become unused. Loading uses `strict=False` where applicable; the loader should **report** mismatched
  keys (missing/unexpected) rather than silently reshaping weights. No unrelated parameters are renamed.

## Risks / Trade-offs

- [Checkpoints trained with `USE_ATTENTION=True` contain `attention.*` keys] → Report missing/unexpected keys explicitly during verification; do not modify those weights.
- [Removal of `attention.py`/`gradcam.py` breaks external imports] → Grep confirms only in-repo consumers (DAMamba, generator, CLI); all are removed in this change.
- [A missed config reference to `USE_ATTENTION` fails config load] → Sweep `configs/` and `config.py`; verify model construction by instantiating `DAMamba_T/S/B`.
- [Unintended change to dynamic scan] → Only touch SE/attention lines; the verification suite (`test_debug_pipeline.py`) and a dummy forward pass confirm Dynamic_Adaptive_Scan still executes.
- [Dead LADAS scaffolding may appear broken] → Document that the generator is intentionally removed pending the next change; the training aux-loss path is gated by `LADAS.ENABLE` (default False) so standard flow is unaffected.