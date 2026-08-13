## ADDED Requirements

### Requirement: Early stopping on validation accuracy plateau
The training script SHALL monitor validation top-1 accuracy (`acc1`) each epoch and stop the training loop after `TRAIN.PATIENCE` consecutive epochs without improvement over the best observed validation accuracy.

#### Scenario: Patience exceeded stops training
- **WHEN** validation `acc1` shows no improvement over `max_accuracy` for `TRAIN.PATIENCE` consecutive epochs
- **THEN** the training loop exits before reaching `TRAIN.EPOCHS`
- **AND** a log message reports the patience counter, the epoch where the best accuracy was reached, and the best validation accuracy

#### Scenario: No improvement but patience not yet reached
- **WHEN** validation `acc1` shows no improvement but fewer than `TRAIN.PATIENCE` consecutive epochs have elapsed since the last improvement
- **THEN** training continues to the next epoch

### Requirement: Patience resets on new best accuracy
The patience counter SHALL reset to zero whenever validation `acc1` sets a new best, i.e. when the existing `max_accuracy` update path fires, and the epoch of the best accuracy SHALL be recorded.

#### Scenario: New best accuracy resets patience
- **WHEN** validation `acc1` exceeds `max_accuracy`
- **THEN** `max_accuracy` is updated and the patience counter resets to zero
- **AND** `best_ckpt.pth` is saved using the existing behavior

### Requirement: Configurable patience
The patience value SHALL be configurable via `TRAIN.PATIENCE` in the config, defaulting to 10. The maximum number of training epochs SHALL remain `TRAIN.EPOCHS` unchanged.

#### Scenario: Default patience value
- **WHEN** `TRAIN.PATIENCE` is not set in the config YAML
- **THEN** the default value of 10 is used

#### Scenario: Maximum epochs unchanged
- **WHEN** early stopping does not trigger
- **THEN** training still runs up to `TRAIN.EPOCHS`

### Requirement: Resume-safe early stopping state
The early-stopping state (patience counter and best-accuracy epoch) SHALL be preserved when resuming from a checkpoint, and SHALL NOT reset the persisted best-accuracy state. Checkpoint files SHALL remain backward-compatible: new keys are optional and old checkpoints without them load correctly.

#### Scenario: Resume preserves patience and best accuracy
- **WHEN** training resumes from a checkpoint that contains early-stopping state
- **THEN** the restored patience counter and best-accuracy epoch continue from the saved values
- **AND** `max_accuracy` is restored as before

#### Scenario: Resume from checkpoint without early-stopping state
- **WHEN** training resumes from a checkpoint that has no early-stopping keys
- **THEN** patience initializes to zero and `max_accuracy` is still restored from the checkpoint

### Requirement: Existing behavior preserved
The optimizer, LR scheduler, model architecture, EMA, dataset handling, and existing checkpoint key formats SHALL be unchanged by the early-stopping feature.

#### Scenario: Best checkpoint saving preserved
- **WHEN** a new best validation accuracy is achieved
- **THEN** `best_ckpt.pth` is saved with the same contents and behavior as before the change

#### Scenario: Post-training evaluation still runs
- **WHEN** training ends (whether by early stopping or reaching `TRAIN.EPOCHS`)
- **THEN** the post-training test evaluation and experiment tracker finalization run as before
