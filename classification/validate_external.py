#!/usr/bin/env python3
"""
External Validation Script - validate a trained DAMamba checkpoint on an
external folder (val or test) that is NOT part of the split.

Usage:
  # External val (when data was split train/test only):
  python classification/validate_external.py --checkpoint output/tiny/finetune/best_ckpt.pth --val-data-path /path/to/external_val
  # External test (when data was split train/val only - your current case):
  python classification/validate_external.py --checkpoint output/tiny/finetune/best_ckpt.pth --test-data-path /path/to/external_test
  python validate.py --checkpoint <ckpt> --test-data-path /path/to/external_test

The external folder must be an ImageFolder with class subfolders:
  /path/to/external_test/
    class_A/
    class_B/
    class_C/
    class_D/

Examples:
  1) Train/val split + external test (your case):
     python -m classification.utils.split_dataset /data/source --train-ratio 0.8 --val-ratio 0.2 --test-ratio 0.0
     -> creates /data/source_split/train and /data/source_split/val
     python -m classification.main --cfg classification/configs/DAMamba/damamba_tiny.yaml --data-path /data/source_split --test-data-path /path/to/external_test

  2) Train/test split + external val:
     python -m classification.utils.split_dataset /data/source --train-ratio 0.8 --val-ratio 0 --test-ratio 0.2
     python -m classification.main --cfg ... --data-path /data/source_split --val-data-path /path/to/external_val
"""

import os
import sys

# Allow running as `python classification/validate_external.py` or `python -m classification.validate_external`
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'classification'))

# Reuse the main validation logic from root validate.py
# Import and delegate so we don't duplicate code
if __name__ == '__main__':
    # Rewrite args to map --val-data-path -> --dataset-path for underlying script
    # Just exec validate.py with the same argv
    validate_path = os.path.join(root_dir, 'validate.py')
    # Pass through all args; validate.py now understands --val-data-path directly
    import subprocess
    cmd = [sys.executable, validate_path] + sys.argv[1:]
    print(f"Delegating to validate.py: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))
