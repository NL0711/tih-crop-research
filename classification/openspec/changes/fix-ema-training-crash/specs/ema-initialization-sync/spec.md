## ADDED Requirements

### Requirement: EMA initialized from loaded pretrained model weights
After `load_pretrained_ema` completes, the system SHALL synchronize the EMA model's parameters and buffers with the freshly-loaded student model (`model_ema.ema.load_state_dict(model_without_ddp.state_dict())`) so the EMA begins training identical to the 5-class pretrained model. This MUST NOT alter `load_pretrained_ema` itself or the pretrained checkpoint loading semantics.

#### Scenario: Pretrained model achieves 80.7% pre-training
- **WHEN** the student model evaluates at ~80.7% on the validation set immediately after loading pretrained weights
- **THEN** the EMA model evaluated with the same weights also reports the same accuracy (not 0%)

#### Scenario: State dict shapes match
- **WHEN** the student and EMA models share the same architecture (5-class head, hybrid CNN/fusion modules)
- **THEN** the sync copies every matching key without dropping or raising

### Requirement: EMA update behavior unchanged during training
The system SHALL preserve the online EMA update (`model_ema.update(model)` at the gradient-accumulation boundary) so the EMA continues to track the training ensemble exactly as before; the sync is a one-time startup action, not a training-loop change.

#### Scenario: EMA tracks model during training
- **WHEN** an optimizer step completes
- **THEN** `model_ema.update(model)` is invoked and EMA weights reflect the exponentially-weighted moving average

#### Scenario: No interference with MESA
- **WHEN** MESA distillation is active
- **THEN** the one-time startup sync does not repeat and does not override the EMA mid-training