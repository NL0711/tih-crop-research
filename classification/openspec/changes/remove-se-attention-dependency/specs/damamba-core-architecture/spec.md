## ADDED Requirements

### Requirement: Vanilla DAMamba forward flow without SE/attention
The DAMamba model SHALL run the classification feature path as
`stem → stage 1 → norm1 → stage 2 → norm2 → stage 3 → norm3 → stage 4 → norm4 → classifier`,
with no SE/attention module applied to the stage feature maps.

#### Scenario: Forward pass yields final normalized stage feature
- **WHEN** a batch of images is passed to `DAMamba.forward_features`
- **THEN** the stages and their corresponding `norm{i+1}` are applied in order
- **AND** the returned tensor is the output of the final norm (norm4) without any stage-attention wrapper applied

#### Scenario: Logits have expected shape
- **WHEN** a batch of shape `[B, C, H, W]` is passed through `DAMamba.forward`
- **THEN** logits have shape `[B, num_classes]`

#### Scenario: No attention modules exist in the model
- **WHEN** the model is instantiated with default `use_attention`/`use_se_only`-free configuration
- **THEN** `isinstance(module, StageAttentionWrapper)` is never true for any submodule
- **AND** no `attention.*` submodule state exists

### Requirement: No SE/attention configuration surface
The model configuration SHALL NOT expose `USE_ATTENTION` or `USE_SE_ONLY` options, and model
construction SHALL NOT require or read them.

#### Scenario: Model builds without attention options
- **WHEN** `build_model(config)` is called with a YAML config that contains no `USE_ATTENTION`/`USE_SE_ONLY` keys
- **THEN** a `DAMamba` model is constructed successfully with no `attention` module

#### Scenario: Legacy config keys are absent
- **WHEN** the default config is loaded
- **THEN** `config.MODEL.DAMAMBA` does not define `USE_ATTENTION` or `USE_SE_ONLY`

### Requirement: Dynamic_Adaptive_Scan preserved
The `Dynamic_Adaptive_Scan`, `DASSM`, `DCNv3`, offset prediction/initialization, multi-group sampling,
`stage_idx`/`block_idx` assignment, and scan-debug functionality SHALL remain functionally identical
after the SE/attention removal.

#### Scenario: Dynamic scan still executes
- **WHEN** a forward pass runs with scan debug enabled
- **THEN** debug records are produced with the same structure (stage, block, offsets, grids, scans) as before the change

#### Scenario: Scan debug does not alter inference
- **WHEN** the same input runs with debug disabled and with debug enabled
- **THEN** the outputs are equal within numerical tolerance

### Requirement: Baseline checkpoint compatibility
Loading a baseline (vanilla, no-attention) DAMamba checkpoint SHALL remain possible, and any
checkpoint keys that become unused after removing attention SHALL be reported rather than silently
modified.

#### Scenario: Vanilla checkpoint loads cleanly
- **WHEN** a checkpoint produced by a no-attention DAMamba is loaded into the updated model
- **THEN** there are no missing keys for shared weights

#### Scenario: Attention-trained checkpoint keys are reported
- **WHEN** a checkpoint that contains `attention.attention_modules.*` keys is loaded with strict checking disabled
- **THEN** the unused `attention.*` keys are reported as unexpected/incompatible
- **AND** no unrelated weight names are renamed or overwritten

### Requirement: SEGradCAM removed from LADAS pipeline
The LADAS pipeline SHALL NOT depend on `SEGradCAM` or SE-specific features for model building or
training. The SEGradCAM-based pseudo-lesion generator SHALL be removed without being replaced.

#### Scenario: Model import does not require SEGradCAM
- **WHEN** the DAMamba model or build pipeline is imported
- **THEN** no module imports `SEGradCAM` or `SEBlock`

#### Scenario: No SE-based generator runs
- **WHEN** the repository is scanned for the removed generator
- **THEN** `utils/generate_pseudo_lesions.py` no longer seeds generation from `SEGradCAM`