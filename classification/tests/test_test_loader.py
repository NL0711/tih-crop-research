import os
import tempfile
import shutil
from pathlib import Path

import torch
from torchvision import datasets

from data.build import build_dataset
from config import get_config


class DummyArgs:
    def __init__(self):
        self.cfg = 'configs/DAMamba/damamba_tiny.yaml'
        self.opts = None
        self.batch_size = None
        self.data_path = None
        self.zip = False
        self.cache_mode = None
        self.pretrained = None
        self.resume = None
        self.accumulation_steps = None
        self.use_checkpoint = False
        self.disable_amp = False
        self.output = './output'
        self.tag = 'test'
        self.oversample = False
        self.eval = False
        self.throughput = False
        self.traincost = False
        self.enable_persistance = False
        self.enable_amp = False
        self.fused_layernorm = False
        self.optim = None
        self.ddp = 'torch'


def _write_image(path: Path):
    import PIL.Image as Image

    img = Image.new('RGB', (32, 32), color=(255, 0, 0))
    img.save(path)


def test_build_dataset_can_use_test_split(tmp_path):
    root = tmp_path / 'dataset'
    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            (root / split / cls).mkdir(parents=True, exist_ok=True)
            _write_image(root / split / cls / f'{split}_{cls}_1.png')
            _write_image(root / split / cls / f'{split}_{cls}_2.png')

    args = DummyArgs()
    args.data_path = str(root)
    config = get_config(args)
    config.defrost()
    config.DATA.DATASET = 'imagenet'
    config.freeze()

    train_ds, _ = build_dataset(is_train=True, config=config)
    assert train_ds is not None
    val_ds, _ = build_dataset(is_train=False, config=config)
    assert val_ds is not None
    assert len(train_ds) == 4
    assert len(val_ds) == 4

    test_ds, _ = build_dataset(is_train=False, config=config, split='test')
    assert test_ds is not None
    assert len(test_ds) == 4
    assert isinstance(test_ds, datasets.ImageFolder)
