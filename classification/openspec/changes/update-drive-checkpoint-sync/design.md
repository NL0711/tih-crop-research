## Context

The tracker (`classification/utils/experiment_tracker.py`) runs on Colab and mirrors experiment artifacts to Google Drive. Two problems exist:

1. `_mount_drive()` unconditionally calls `drive.mount()` even when Drive is already mounted at `/content/drive`, forcing a redundant authorization flow / mount operation on each run.
2. `sync_checkpoint_to_drive()` delegates to `_sync_file()`, which on any exception only logs `log.warning(...)` and returns. A failed checkpoint sync is invisible: no source/destination logging, no verification, no error surfaced.

Checkpoint syncs originate from `main.py` at `latest_ckpt` (line ~339), `best_ckpt` (line ~357), and `best_ckpt_ema` (line ~377). The tracker already maintains `drive_exp_dir = DRIVE_ROOT/<exp_name>/<timestamp>/` with a `checkpoints/` subdirectory created in `_setup_dirs()`.

## Goals / Non-Goals

**Goals:**
- Make Drive mount idempotent: skip `drive.mount()` when `/content/drive` is already mounted.
- For every checkpoint sync, log exact local source and Drive destination.
- Copy with error handling that never silently fails for checkpoints.
- Verify destination exists after copy and log its size.
- Raise clear errors on copy/verification failure.
- Minimal, localized changes to `experiment_tracker.py`.

**Non-Goals:**
- Changing checkpoint naming, training loop, or experiment-tracking behavior.
- Altering the sync behavior of non-checkpoint files/directories (CSV, JSON, plots, dir mirror) beyond the idempotent mount change.
- Adding new dependencies.

## Decisions

### 1. Idempotent Drive mount via `os.path.isdir` check
Detect an existing mount by checking `os.path.isdir(mount_point)` (optionally combined with the presence of `MyDrive` subdir). Only call `drive.mount()` when the path is not already mounted.

- **Why this over alternatives:** `drive.mount()` has `force_remount=False` as default already, but it still goes through the auth flow and can error when Drive is already mounted. Checking the filesystem is cheap and reliable on Colab where `/content/drive` exists only when mounted.
- **Alternative considered:** Wrapping `drive.mount()` in try/except and treating failure as "already mounted." Rejected: a genuine auth failure would be misclassified as success, and the current behavior already swallows mount errors.
- **Alternative considered:** Calling `drive.mount()` with `force_remount=False` and checking output. Rejected: no clean programmatic signal from the mount call.

### 2. Hardened checkpoint sync path
`sync_checkpoint_to_drive()` will perform a dedicated, strict sync for checkpoints instead of reusing the permissive `_sync_file()`:

1. Build destination as `drive_exp_dir/checkpoints/<basename>`.
2. Log source and destination before copying.
3. Copy with `shutil.copy2`; on exception raise `RuntimeError` naming both paths.
4. After copy, `os.path.isfile(dest)`; if missing, raise `RuntimeError`.
5. `os.path.getsize(dest)`; if `0`, raise `RuntimeError`; otherwise log the size.

- **Why a new path over modifying `_sync_file()`:** `_sync_file()` is also used by non-checkpoint artifacts (CSV/JSON/report/plots) where the current warning-and-return is acceptable and relied upon (e.g., Drive dir creation may fail in `_setup_dirs`, leaving `drive_exp_dir` set but the checkpoint dir unwritable). Raising there would change behavior for unrelated artifacts. Keeping the strict semantics inside `sync_checkpoint_to_drive()` confines the change to the requested scope.
- **Why `raise`:** The requirement says "Raise/log a clear error"; raising from `sync_checkpoint_to_drive()` surfaces the failure to `main.py` at the call site so the user sees it, while `log.error` still records it. A pure warning would preserve the current "silent-ish" failure mode the change is meant to eliminate.
- **Alternative considered:** Letting `_sync_file` accept a `strict` flag. Rejected as more invasive; the flag would need threading through existing call sites.

### 3. Keep `_sync_file`/`_mirror_dir` otherwise unchanged
Only the mount detection and the checkpoint sync path change. `_sync_file` keeps its warning behavior for non-checkpoint files; `_mirror_dir` and directory layout are untouched.

## Risks / Trade-offs

- [`/content/drive` exists but Drive is in an unusable state] → Mitigation: destination write/verification errors still surface via the strict checkpoint sync error handling.
- [`drive_exp_dir` is `None` (Drive disabled or mount failed)] → Mitigation: `sync_checkpoint_to_drive()` no-ops without error, matching existing behavior (nothing to sync to). Log at debug level.
- [Checkpoint missing locally (e.g., `best_ckpt_ema` not produced)] → Mitigation: existing guard in callers; if a sync is requested for a missing source, raise a clear error rather than silently skip, since silent skip would violate "never silently fail."
- [Raising errors could interrupt training if sync is called mid-training] → Mitigation: syncs are already called post-`save_checkpoint` in `main.py`; a failure is a real data-loss signal and should be loud. No retry logic added (out of scope).

## Migration Plan

No data migration. This is a code-only change to `experiment_tracker.py`; existing experiment directories and Drive layout are unchanged. Rollback is a revert of the single file change. Local (non-Colab) runs are unaffected because `_mount_drive` is only invoked when `in_colab` is true, and checkpoint sync already no-ops when `drive_exp_dir` is `None`.

## Open Questions

None. Scope is fully specified by the proposal and specs.
