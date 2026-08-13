## Why

On Colab, `_mount_drive()` calls `drive.mount()` on every run even when Drive is already mounted at `/content/drive`, which prompts an authorization flow or errors and wastes time. Checkpoint syncs to Drive currently fail silently (logs a warning and returns), so a failed copy goes unnoticed and no record of the source/destination is kept, making it impossible to trust Drive mirror copies of `latest_ckpt`, `best_ckpt`, and `best_ckpt_ema`.

## What Changes

- `_mount_drive()` becomes idempotent: it detects an existing mount at `/content/drive` and skips `drive.mount()` entirely when already mounted.
- Checkpoint sync is hardened. `sync_checkpoint_to_drive()` (and its underlying `_sync_file` path for checkpoints) will:
  - Log the exact local source path and Drive destination path for every sync.
  - Raise a clear error on copy failure (no more silent warning-only behavior for checkpoints).
  - Verify the destination file exists after copying and log its size.
  - Raise a clear error if verification fails (missing destination or zero-size destination).
- Existing experiment-specific directory structure, checkpoint naming, and all other experiment-tracking behavior are unchanged.
- No behavior change for non-checkpoint file/dir sync paths beyond making Drive mount detection idempotent.

## Capabilities

### New Capabilities
- `drive-checkpoint-sync`: Reliable, verifiable checkpoint synchronization from the local experiment directory to the per-experiment Google Drive folder, including idempotent Drive mount detection.

### Modified Capabilities
<!-- None - no existing specs in openspec/specs/ -->

## Impact

- `classification/utils/experiment_tracker.py` (functions `_mount_drive`, `_sync_file`, `sync_checkpoint_to_drive`).
- Callers in `classification/main.py` (lines ~339, ~357, ~377) pass checkpoint paths to `tracker.sync_checkpoint_to_drive(...)`; their call sites and checkpoint naming remain unchanged.
- Requires Colab environment to exercise Drive behavior; local runs are unaffected (Drive sync already no-ops when not in Colab or Drive is disabled).
