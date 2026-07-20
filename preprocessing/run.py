"""
original folder is never modified

Usage:
    python preprocess_folder.py --input PATHNAME --output PATHNAME
"""

import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

try:
    import albumentations as A
except ImportError:
    print("Missing dependency: pip install albumentations --break-system-packages")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DEFAULTS = dict(
    brightness_limit=0.2,
    contrast_limit=0.2,
    sharpen_alpha=(0.2, 0.5),
    sat_limit=30,
    hue_limit=20,
    val_limit=20,
    gaussian_noise_var=(10.0, 50.0),
    gaussian_blur_limit=(3, 7),
    median_blur_limit=(3, 7),
    motion_blur_limit=(3, 15),
    salt_pepper_prob=0.02,
    poisson_scale=1.0,
)


class AdvancedImageEnhancer:
    def __init__(self, params, is_train=True):
        self.is_train = is_train
        self.salt_pepper_prob = params["salt_pepper_prob"]
        self.poisson_scale = params["poisson_scale"]

        basic_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=params["brightness_limit"],
                contrast_limit=params["contrast_limit"],
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Sharpen(alpha=params["sharpen_alpha"], lightness=(0.5, 1.5), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=params["hue_limit"],
                sat_shift_limit=params["sat_limit"],
                val_shift_limit=params["val_limit"],
                p=1.0
            )
        ]
        self.basic_pipeline = A.Compose(basic_transforms)

        if self.is_train:
            advanced_transforms = [
                A.OneOf([
                    A.GaussNoise(var_limit=params["gaussian_noise_var"], p=0.5),
                    A.Lambda(image=self.add_poisson_noise, p=0.3),
                    A.Lambda(image=self.add_salt_and_pepper, p=0.2),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=params["gaussian_blur_limit"], p=0.4),
                    A.MedianBlur(blur_limit=params["median_blur_limit"], p=0.3),
                    A.MotionBlur(blur_limit=params["motion_blur_limit"], p=0.3),
                ], p=0.5)
            ]
            self.advanced_pipeline = A.Compose(advanced_transforms)
        else:
            self.advanced_pipeline = None

    def add_poisson_noise(self, image, **kwargs):
        image_f = image.astype(np.float32)
        noisy = np.random.poisson(image_f * self.poisson_scale) / self.poisson_scale
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_salt_and_pepper(self, image, **kwargs):
        noisy = image.copy()
        h, w, c = image.shape
        num_salt = np.ceil(self.salt_pepper_prob * image.size / 6.0)
        coords_y = np.random.randint(0, h, int(num_salt))
        coords_x = np.random.randint(0, w, int(num_salt))
        noisy[coords_y, coords_x, :] = 255

        num_pepper = np.ceil(self.salt_pepper_prob * image.size / 6.0)
        coords_y = np.random.randint(0, h, int(num_pepper))
        coords_x = np.random.randint(0, w, int(num_pepper))
        noisy[coords_y, coords_x, :] = 0
        return noisy

    def apply_noise_reduction(self, image):
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image.convert("RGB"))
        img_np = self.apply_noise_reduction(img_np)

        augmented = self.basic_pipeline(image=img_np)
        img_np = augmented["image"]

        if self.is_train and self.advanced_pipeline is not None:
            augmented_adv = self.advanced_pipeline(image=img_np)
            img_np = augmented_adv["image"]

        return Image.fromarray(img_np)


def find_images(folder):
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, folder)
                yield abs_path, rel_path


def _process_one(paths):
    src_path, dst_path = paths
    enhancer = AdvancedImageEnhancer(DEFAULTS, is_train=True)
    try:
        with Image.open(src_path) as img:
            out_img = enhancer(img)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        out_img.save(dst_path, quality=95)
        return True, src_path, None
    except Exception as e:
        return False, src_path, str(e)


def main():
    parser = argparse.ArgumentParser(description="Create an enhanced copy of a folder of images.")
    parser.add_argument("--input", required=True, help="Folder to read images from (never modified)")
    parser.add_argument("--output", required=True, help="Folder to write the enhanced copy to")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"Input folder does not exist: {input_dir}")
        sys.exit(1)
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        print("Output folder must be different from the input folder.")
        sys.exit(1)

    tasks = [
        (src, os.path.join(output_dir, rel))
        for src, rel in find_images(input_dir)
    ]

    if not tasks:
        print(f"No images found under {input_dir}")
        sys.exit(0)

    print(f"Found {len(tasks)} images in {input_dir}")
    print(f"Writing copies to {output_dir} using {args.workers} workers...")

    if args.workers <= 1:
        results = [_process_one(t) for t in tqdm(tasks)]
    else:
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks)))

    failures = [(p, e) for ok, p, e in results if not ok]
    print(f"Done. {len(tasks) - len(failures)}/{len(tasks)} images processed successfully.")
    if failures:
        print(f"{len(failures)} images failed:")
        for path, err in failures[:20]:
            print(f"  {path}: {err}")

    print(f"\copy available at: {output_dir}")


if __name__ == "__main__":
    main()