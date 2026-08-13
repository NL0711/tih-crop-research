## ADDED Requirements

### Requirement: MESA soft-target distillation loss uses KL divergence
When `mesa > 0.0`, the system SHALL compute `ema_loss` as a soft-target distillation term between the student model's log-probabilities and the EMA teacher's detached softmax probabilities (KL divergence), then multiply by `mesa`. The EMA teacher forward MUST run under `torch.inference_mode()` and its output MUST be detached before use. The system MUST NOT pass float probability tensors as a class-index target to a CrossEntropy-family criterion, and MUST NOT cast `ema_output` to `long`.

#### Scenario: MESA active (crash regression)
- **WHEN** `mesa > 0.0` and the EMA model produces a `[B, 5]` float probability tensor from a 5-class model
- **THEN** `ema_loss = KL(student_log_softmax, ema_softmax.detach()) * mesa` is computed without raising `gather(): Expected dtype int64 for index`

#### Scenario: Critertion type is independent of the distillation term
- **WHEN** the classification criterion is `torch.nn.CrossEntropyLoss`, `timm.LabelSmoothingCrossEntropy`, or `timm.SoftTargetCrossEntropy`
- **THEN** the distillation term still uses its own KL divergence and never relies on the classification criterion accepting a float target

#### Scenario: Total loss composition
- **WHEN** `mesa > 0.0`
- **THEN** the training loss equals `criterion(outputs, targets) + ema_loss`, with `ema_loss` scaled by `mesa`, and no other loss terms are added

### Requirement: EMA update and evaluation preserved
The system SHALL keep the existing EMA update step (`model_ema.update(model)` after each optimizer step) and the existing EMA evaluation calls unchanged. The MESA distillation change MUST affect only the training loss term.

#### Scenario: EMA continues to track the trained model
- **WHEN** training runs with the corrected loss
- **THEN** `model_ema.update(model)` is still invoked on the gradient-accumulation boundary and EMA checkpoints are still saved/loaded as before

## REMOVED Requirements

### Requirement: Class-index criterion applied to EMA probabilities
**Reason**: The original code `ema_loss = criterion(outputs, ema_output) * mesa` fed a float probability tensor to a class-index criterion, crashing with `gather(): Expected dtype int64 for index`.
**Migration**: Use the new KL-divergence soft-target term from `ema-soft-target-distillation`.