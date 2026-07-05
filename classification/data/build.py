# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler
from utils.distributed import get_rank, get_world_size, is_dist_avail_and_initialized
from collections import Counter, defaultdict
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def _oversample_dataset(dataset, seed=0, replace=True):
    labels = getattr(dataset, 'targets', None)
    if labels is None:
        labels = getattr(dataset, 'labels', None)

    samples = getattr(dataset, 'samples', None)
    if samples is None:
        samples = getattr(dataset, 'imgs', None)

    if labels is None or samples is None:
        raise ValueError('Oversampling requires dataset with `samples` and `targets`/`labels` attributes.')

    if len(labels) != len(samples):
        raise ValueError('Dataset samples and labels lengths do not match for oversampling.')

    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    max_count = max(len(idx_list) for idx_list in class_indices.values())
    if max_count == 0:
        return dataset

    rng = np.random.RandomState(seed)
    extra_samples = []
    extra_labels = []
    for label, idx_list in class_indices.items():
        current_count = len(idx_list)
        if current_count >= max_count:
            continue
        choose_replace = replace or (max_count - current_count > len(idx_list))
        choices = rng.choice(idx_list, size=max_count - current_count, replace=choose_replace)
        for choice in choices:
            extra_samples.append(samples[choice])
            extra_labels.append(labels[choice])

    if extra_samples:
        dataset.samples = list(samples) + extra_samples
        if hasattr(dataset, 'imgs'):
            dataset.imgs = list(samples) + extra_samples
        if hasattr(dataset, 'targets'):
            dataset.targets = list(labels) + extra_labels
        elif hasattr(dataset, 'labels'):
            dataset.labels = list(labels) + extra_labels

    return dataset


def _print_dataset_distribution(dataset, name):
    labels = getattr(dataset, 'targets', None)
    if labels is None:
        labels = getattr(dataset, 'labels', None)
    if labels is None:
        return

    counts = Counter(labels)
    classes = getattr(dataset, 'classes', None)
    print(f"{name} class distribution (samples={len(labels)}):")
    print("| class | idx | count |")
    for idx in sorted(counts.keys()):
        class_name = classes[idx] if classes is not None and idx < len(classes) else str(idx)
        print(f"| {class_name} | {idx} | {counts[idx]} |")


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    _print_dataset_distribution(dataset_train, 'Train (before oversampling)')
    if config.DATA.OVERSAMPLE:
        dataset_train = _oversample_dataset(
            dataset_train,
            seed=config.SEED,
            replace=config.DATA.OVERSAMPLE_REPLACEMENT,
        )
        print(f"rank {get_rank()} enabled class-balanced oversampling; training samples = {len(dataset_train)}")
        _print_dataset_distribution(dataset_train, 'Train (after oversampling)')
    config.freeze()
    print(f"rank {get_rank()} successfully build train dataset")

    dataset_val, _ = build_dataset(is_train=False, config=config, split='val')
    _print_dataset_distribution(dataset_val, 'Validation')
    print(f"rank {get_rank()} successfully build val dataset")

    dataset_test = None
    data_loader_test = None
    test_root = os.path.join(config.DATA.DATA_PATH, 'test')
    if os.path.isdir(test_root):
        dataset_test, _ = build_dataset(is_train=False, config=config, split='test')
        _print_dataset_distribution(dataset_test, 'Test')
        print(f"rank {get_rank()} successfully build test dataset")

    num_tasks = get_world_size()
    global_rank = get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(get_rank(), len(dataset_train), get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    elif num_tasks > 1:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = RandomSampler(dataset_train)

    if config.TEST.SEQUENTIAL or num_tasks <= 1:
        sampler_val = SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    if config.MODEL.DDP == 'torch':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

        if dataset_test is not None:
            sampler_test = SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False
            )
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


def build_dataset(is_train, config, split='val'):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else split
        try:
            ddp = config.MODEL.DDP
        except:
            ddp = 'torch'

        if ddp == 'torch':
            if config.DATA.ZIP_MODE:
                ann_file = prefix + "_map.txt"
                prefix = prefix + ".zip@/"
                dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                            cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
            else:
                root = os.path.join(config.DATA.DATA_PATH, prefix)
                if not os.path.isdir(root):
                    if not is_train and split == 'test':
                        return None, 0
                    raise FileNotFoundError(f"Dataset folder not found: {root}")
                dataset = datasets.ImageFolder(root, transform=transform)

    # =============================================================================
    # # JUST for test
                if False:
                    from torch.utils.data import Dataset
                    class FDataset(Dataset):
                        def __init__(self, *args, **kwargs):
                            pass

                        def __len__(self):
                            return 1000

                        def __getitem__(self, *args,**kwargs):
                            return torch.randn((3, 224, 224)), 0

                    dataset = FDataset()

# =============================================================================

        nb_classes = len(dataset.classes) if dataset is not None else 0
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
