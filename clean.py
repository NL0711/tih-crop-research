import argparse
from multiprocessing import Pool
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class AdvancedImageEnhancer:
    """
    Deterministic image enhancement pipeline:

    1. Noise Reduction (Bilateral Filter)
    2. Contrast Adjustment
    3. Histogram Equalization (CLAHE)
    4. Sharpening
    """

    def __init__(
        self,
        contrast=1.15,
        clahe_clip=2.0,
        clahe_grid=(4, 4)
    ):
        self.contrast = contrast
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    def apply_noise_reduction(self, image):
        return cv2.bilateralFilter(
            image,
            d=5,
            sigmaColor=50,
            sigmaSpace=75
        )

    def apply_brightness_contrast(self, image):
        return cv2.convertScaleAbs(
            image,
            alpha=self.contrast,
        )

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_grid
        )

        l = clahe.apply(l)

        enhanced = cv2.merge((l, a, b))

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_LAB2RGB
        )

    def apply_sharpen(self, image):
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ], dtype=np.float32)

        return cv2.filter2D(
            image,
            ddepth=-1,
            kernel=kernel
        )

    def __call__(self, image):
        img = np.array(image.convert("RGB"))

        img = self.apply_noise_reduction(img)
        img = self.apply_brightness_contrast(img)
        img = self.apply_clahe(img)
        img = self.apply_sharpen(img)

        return Image.fromarray(img)

def find_images(root_dir):
    image_extensions = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".tif", ".tiff",
        ".webp"
    }

    for root, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext in image_extensions:
                src_path = os.path.join(root, file)

                rel_path = os.path.relpath(
                    src_path,
                    root_dir
                )

                yield src_path, rel_path

def _process_one(paths):
    src_path, dst_path = paths

    enhancer = AdvancedImageEnhancer(
        contrast=1.15,
        clahe_clip=3.0
    )

    try:
        with Image.open(src_path) as img:
            out_img = enhancer(img)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        ext = os.path.splitext(dst_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            out_img.save(dst_path, quality=95)
        else:
            out_img.save(dst_path)

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

    print(f"\nCopy available at: {output_dir}")


if __name__ == "__main__":
    main()