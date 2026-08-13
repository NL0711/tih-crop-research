## ADDED Requirements

### Requirement: Hybrid flag propagated to model construction
The system SHALL forward the existing `config.MODEL.HYBRID.ENABLE` value into the `DAMamba` constructor as `hybrid_enable=...` in `build_model()`, using the exact configuration key already defined in `config.py`. The model SHALL NOT invent new configuration keys.

#### Scenario: Hybrid enabled
- **WHEN** `config.MODEL.HYBRID.ENABLE` is `True`
- **THEN** `DAMamba(..., hybrid_enable=True)` is invoked and the model instantiates `local_cnn` and `fusion`

#### Scenario: Hybrid disabled
- **WHEN** `config.MODEL.HYBRID.ENABLE` is `False`
- **THEN** `DAMamba(..., hybrid_enable=False)` is invoked and the model follows the original vanilla forward path with no `local_cnn` or `fusion` modules

### Requirement: Hybrid forward path unchanged
The system SHALL preserve the existing `DAMamba` internal architecture — `LocalCNN`, `HybridFusion`, `Dynamic_Adaptive_Scan`, DCNv3, and selective-scan — exactly as-is. The implementation SHALL only change how the model is constructed, not how the hybrid branches compute.

#### Scenario: Hybrid construction preserves architecture
- **WHEN** a hybrid-enabled model is constructed
- **THEN** its CNN, fusion, scanning, and attention components match the reference implementation

### Requirement: Backward compatibility when hybrid disabled
The system SHALL produce byte-identical behavior between the corrected code and the previous code when `MODEL.HYBRID.ENABLE` is `False`, so existing vanity/model evaluation results remain reproducible.

#### Scenario: Disabled hybrid behaves as before
- **WHEN** a model is constructed with the hybrid flag disabled using the corrected `build_model`
- **THEN** its forward output matches the output of a model constructed by the previous `build_model`