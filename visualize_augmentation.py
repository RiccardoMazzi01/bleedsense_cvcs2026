import argparse
import os
import random

import albumentations as A
import cv2
import numpy as np

from dataset import load_hemoset_data

PANEL_SIZE = 256
LABEL_HEIGHT = 24


def get_visual_aggressive_transform(img_size=(256, 256)):
    """Same pixel-level operations as get_aggressive_transforms() in dataset.py, but
    without Normalize/ToTensorV2, so the output stays a viewable uint8 RGB image.
    """
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.RandomGamma(p=0.4),
        A.OneOf([A.GaussianBlur(), A.MotionBlur()], p=0.3),
        A.GaussNoise(p=0.3),
        A.ImageCompression(p=0.3),
    ])


def add_label(panel, text):
    labeled = np.full((PANEL_SIZE + LABEL_HEIGHT, PANEL_SIZE, 3), 255, dtype=np.uint8)
    labeled[LABEL_HEIGHT:, :, :] = panel
    cv2.putText(labeled, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return labeled


def build_row(image_path, transform, img_size=(256, 256), num_variants=2):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)

    panels = [add_label(image, "original")]
    for i in range(num_variants):
        augmented = transform(image=image)["image"]
        panels.append(add_label(augmented, f"aggressive augmentation #{i + 1}"))

    row = np.concatenate(panels, axis=1)
    return cv2.cvtColor(row, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/work/cvcs2026/bleedsense/datasets")
    parser.add_argument("--output-dir", default="/work/cvcs2026/bleedsense/results/qualitative")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--num-variants", type=int, default=2,
                         help="How many independently-sampled augmented versions to show per example")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    transform = get_visual_aggressive_transform()

    images, masks, _, _ = load_hemoset_data(os.path.join(args.data_dir, "hemoset"))
    non_empty = [i for i in range(len(masks))
                 if (cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE) > 0).sum() > 500]
    random.Random(args.seed).shuffle(non_empty)
    chosen = non_empty[:args.num_examples]

    for i, idx in enumerate(chosen):
        random.seed(args.seed + i)
        row = build_row(images[idx], transform, num_variants=args.num_variants)
        out_path = os.path.join(args.output_dir, f"augmentation_example{i + 1}.png")
        cv2.imwrite(out_path, row)
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
