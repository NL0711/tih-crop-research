## ADDED Requirements

### Requirement: Drive mount detection is idempotent
The system SHALL detect whether Google Drive is already mounted at `/content/drive` before attempting to mount. If Drive is already mounted, the system MUST NOT call `drive.mount()`.

#### Scenario: Drive is already mounted
- **WHEN** the tracker initializes in a Colab environment where Google Drive is already mounted at `/content/drive`
- **THEN** the tracker skips `drive.mount()` and proceeds using the existing mount without prompting for authorization

#### Scenario: Drive is not mounted
- **WHEN** the tracker initializes in a Colab environment where Google Drive is not mounted at `/content/drive`
- **THEN** the tracker mounts Drive at `/content/drive` and proceeds

#### Scenario: Mount attempt fails
- **WHEN** the tracker attempts to mount Drive and the mount fails
- **THEN** the tracker logs a warning and disables the Drive mirror for the run, matching existing behavior

### Requirement: Checkpoint sync logs source and destination
The system SHALL log the exact local source path and the exact Drive destination path for every checkpoint sync of `latest_ckpt`, `best_ckpt`, and `best_ckpt_ema`.

#### Scenario: Checkpoint sync initiated
- **WHEN** a checkpoint is synced to Drive via `sync_checkpoint_to_drive()`
- **THEN** a log entry records the exact local source path and the exact Drive destination path

### Requirement: Checkpoint sync copy has error handling
The system SHALL copy each checkpoint to Drive with error handling and MUST NOT silently fail. If the copy operation raises an exception, the system SHALL raise a clear error identifying the source and destination paths.

#### Scenario: Copy fails
- **WHEN** the copy operation of a checkpoint to Drive raises an exception
- **THEN** the system raises a clear error that includes the source and destination paths

#### Scenario: Copy succeeds
- **WHEN** the copy operation of a checkpoint to Drive completes without exception
- **THEN** the system proceeds to verify the destination file

### Requirement: Checkpoint sync verifies destination
The system SHALL verify the destination file exists after copying and SHALL log the size of the destination file.

#### Scenario: Destination exists after copy
- **WHEN** a checkpoint has been copied to Drive
- **THEN** the system checks that the destination file exists and logs its size in bytes

#### Scenario: Destination missing after copy
- **WHEN** a checkpoint copy completes but the destination file does not exist
- **THEN** the system raises a clear error identifying the missing destination path

#### Scenario: Destination exists but is empty
- **WHEN** a checkpoint copy completes and the destination file exists but has zero size
- **THEN** the system raises a clear error identifying the destination path and its zero size

### Requirement: Experiment directory structure is preserved
The system SHALL keep the existing experiment-specific directory structure on Drive, where each experiment maps to `DRIVE_ROOT/<exp_name>/<timestamp>/` with a `checkpoints/` subdirectory. This change MUST NOT alter checkpoint naming, training behavior, or any other experiment-tracking behavior.

#### Scenario: Sync targets per-experiment checkpoints directory
- **WHEN** a checkpoint is synced to Drive
- **THEN** it is written under the same per-experiment directory (`DRIVE_ROOT/<exp_name>/<timestamp>/checkpoints/`) as before

#### Scenario: No side effects on non-checkpoint behavior
- **WHEN** the change is applied
- **THEN** checkpoint naming, training, and all other experiment-tracking behavior remain unchanged
