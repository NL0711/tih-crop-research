import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def find_image_files(folder: Path, extensions: Iterable[str]) -> List[Path]:
    files = []
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return files


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_list(items: List, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List, List, List]:
    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def copy_subset(files: List[Path], src_root: Path, dest_root: Path) -> List[str]:
    output_paths = []
    for src_path in files:
        relative_class = src_path.parent.relative_to(src_root)
        dest_dir = dest_root / relative_class
        make_dir(dest_dir)
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        output_paths.append(str(dest_path.relative_to(dest_root.parent)))
    return output_paths


def _print_split_summary(summary):
    total_train = sum(item['train'] for item in summary['classes'].values())
    total_val = sum(item['val'] for item in summary['classes'].values())
    total_test = sum(item['test'] for item in summary['classes'].values())
    total_images = sum(item['total'] for item in summary['classes'].values())

    print('\nClass distribution summary:')
    print('| Class | Total | Train | Validation | Test |')
    print('| --- | --- | --- | --- | --- |')
    for class_name, counts in sorted(summary['classes'].items()):
        print(f"| {class_name} | {counts['total']} | {counts['train']} | {counts['val']} | {counts['test']} |")
    print(f"| Total | {total_images} | {total_train} | {total_val} | {total_test} |\n")


def build_split_files(root_dir: Path,
                      output_dir: Path,
                      train_ratio: float,
                      val_ratio: float,
                      test_ratio: float,
                      seed: int,
                      extensions: Iterable[str] = IMAGE_EXTENSIONS,
                      copy_files: bool = True,
                      move_files: bool = False) -> None:
    if move_files and not copy_files:
        raise ValueError('move_files requires copy_files to be True')

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.0')

    root_dir = root_dir.resolve()
    output_dir = output_dir.resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f'Root directory does not exist: {root_dir}')

    random.seed(seed)
    class_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f'No class subfolders found under {root_dir}')

    summary = {
        'root_dir': str(root_dir),
        'output_dir': str(output_dir),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed,
        'classes': {}
    }

    train_paths, val_paths, test_paths = [], [], []
    for class_dir in sorted(class_dirs):
        image_files = find_image_files(class_dir, extensions)
        if not image_files:
            print(f'Warning: no images found in class folder {class_dir}')
            continue

        random.shuffle(image_files)
        train_files, val_files, test_files = split_list(image_files, train_ratio, val_ratio, test_ratio)

        class_name = class_dir.name
        summary['classes'][class_name] = {
            'total': len(image_files),
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
        }

        if copy_files:
            train_paths.extend(copy_subset(train_files, root_dir, output_dir / 'train'))
            val_paths.extend(copy_subset(val_files, root_dir, output_dir / 'val'))
            test_paths.extend(copy_subset(test_files, root_dir, output_dir / 'test'))
        else:
            train_paths.extend([str(p.relative_to(root_dir)) for p in train_files])
            val_paths.extend([str(p.relative_to(root_dir)) for p in val_files])
            test_paths.extend([str(p.relative_to(root_dir)) for p in test_files])

        if move_files:
            for path in image_files:
                path.unlink()

    make_dir(output_dir)
    with open(output_dir / 'train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_paths))
    with open(output_dir / 'val.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_paths))
    with open(output_dir / 'test.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_paths))

    with open(output_dir / 'split_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Split completed:')
    print(f'  train: {len(train_paths)}')
    print(f'  val:   {len(val_paths)}')
    print(f'  test:  {len(test_paths)}')
    print(f'  output: {output_dir}')
    _print_split_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test by class folder')
    parser.add_argument('root_dir', type=Path, help='Dataset root folder. Subfolders are treated as classes.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output folder for train/val/test splits. Default: <root_dir>_split')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--no-copy', action='store_true', help='Do not copy files; only write split text files')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying them into the split directories')
    parser.add_argument('--ext', nargs='+', default=list(IMAGE_EXTENSIONS),
                        help='Image file extensions to include')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root_dir.with_name(f'{args.root_dir.name}_split')
    build_split_files(
        root_dir=args.root_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        extensions={ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in args.ext},
        copy_files=not args.no_copy,
        move_files=args.move,
    )


if __name__ == '__main__':
    main()
