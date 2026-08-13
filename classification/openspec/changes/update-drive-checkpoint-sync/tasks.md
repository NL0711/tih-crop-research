## 1. Idempotent Drive Mount

- [x] 1.1 Update `_mount_drive()` in `utils/experiment_tracker.py` to detect an existing mount at `/content/drive` via `os.path.isdir(mount_point)` and skip `drive.mount()` when already mounted
- [x] 1.2 Keep the existing failure path (log warning, return `False`) when mount is required but fails

## 2. Hardened Checkpoint Sync

- [x] 2.1 Rewrite `sync_checkpoint_to_drive()` to build destination as `drive_exp_dir/checkpoints/<basename>` and log the exact local source and Drive destination before copying
- [x] 2.2 In `sync_checkpoint_to_drive()`, copy with `shutil.copy2` and raise a clear `RuntimeError` naming source and destination if the copy raises
- [x] 2.3 After copying, verify the destination exists (`os.path.isfile`), log its size in bytes, and raise a clear error if missing
- [x] 2.4 Raise a clear error if the destination exists but has zero size; otherwise log the successful sync with size

## 3. Regression Safety

- [x] 3.1 Confirm no change to `_sync_file()` / `_mirror_dir()` behavior for non-checkpoint artifacts and to checkpoint naming / directory structure
- [x] 3.2 Verify the file compiles (`python -m py_compile utils/experiment_tracker.py`) and call sites in `main.py` are unchanged
