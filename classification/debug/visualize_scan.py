"""
Standalone script to visualize Dynamic_Adaptive_Scan behavior for a single image.

Usage:
    uv run python visualize_scan.py --image path/to/image.jpg --output ./scan_viz --model DAMamba_T
    uv run python visualize_scan.py --image path/to/image.jpg --output ./scan_viz --model DAMamba_T --checkpoint path/to/best_ckpt.pth --num-classes 1000

Run this from inside the `classification/` folder (same level as models/, debug/, main.py),
so the relative imports used by DAMamba.py resolve the same way they do at training time.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.DAMamba import DAMamba_T, DAMamba_S, DAMamba_B
from debug.export import export_debug_data
from debug.visualizer import DASVisualizer


MODEL_FNS = {
    "DAMamba_T": DAMamba_T,
    "DAMamba_S": DAMamba_S,
    "DAMamba_B": DAMamba_B,
}


def load_image(image_path, img_size=224):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((img_size, img_size))
    image_np = np.array(img_resized)  # (H, W, 3), uint8

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

    return tensor, image_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Directory to write results to")
    parser.add_argument("--model", default="DAMamba_T", choices=list(MODEL_FNS.keys()))
    parser.add_argument("--checkpoint", default=None, help="Optional path to a .pth checkpoint")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(args.image))[0]

    # 1. Build model
    model_fn = MODEL_FNS[args.model]
    model = model_fn(num_classes=args.num_classes)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)  # supports both raw state_dict and {"model": ...} wrapping
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No checkpoint provided - using randomly initialized weights (offsets will be meaningless).")

    model.to(args.device)
    model.eval()

    # 2. Load image
    input_tensor, image_np = load_image(args.image, img_size=args.img_size)
    input_tensor = input_tensor.to(args.device)

    # 3. Run forward pass with debug recording enabled
    model.enable_scan_debug()
    with torch.no_grad():
        logits = model(input_tensor)

    probs = torch.softmax(logits, dim=-1)
    confidence, predicted_class = probs.max(dim=-1)

    records = model.get_scan_debug_data()
    print(f"Recorded {len(records)} debug entries (one per stage/block Dynamic_Adaptive_Scan call).")

    # 4. Export raw data + metadata
    meta_extra = {
        "predicted_class": int(predicted_class.item()),
        "confidence": float(confidence.item()),
        "true_label": None,       # fill in if you have ground truth for this image
        "correct": None,
        "input_image_size": (args.img_size, args.img_size),
    }
    export_debug_data(args.output, image_name, records, meta_extra)
    print(f"Exported raw arrays + metadata to {os.path.join(args.output, image_name)}")

    # 5. Generate visualization plots
    viz = DASVisualizer(args.output)
    viz.save(image_np, image_name, records)
    print(f"Saved visualization plots to {os.path.join(args.output, image_name)}")

    # 6. Clean up debug state so the model object is safe to reuse for another image
    model.disable_scan_debug()
    model.clear_scan_debug()

    print(f"\nPredicted class: {meta_extra['predicted_class']} (confidence: {meta_extra['confidence']:.4f})")


if __name__ == "__main__":
    main()