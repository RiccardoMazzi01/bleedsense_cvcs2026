import argparse
import os
import random

import cv2
import numpy as np
import torch

from dataset import get_transforms, load_hemoset_data, load_rabbani_data, split_rabbani_data
from models import get_model

SPLIT_SEED = 42
ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]
PANEL_SIZE = 256
LABEL_HEIGHT = 24


def load_checkpoint(architecture, run_name, encoder, device, ckpt_dir):
    model = get_model(architecture, encoder_name=encoder).to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{run_name}_best.pth"), map_location=device))
    model.eval()
    return model


def predict(model, image_tensor, device):
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def predict_ensemble(models, image_tensor, device):
    with torch.no_grad():
        probs = [torch.sigmoid(m(image_tensor.unsqueeze(0).to(device)))[0, 0].cpu().numpy() for m in models]
    return (np.mean(probs, axis=0) > 0.5).astype(np.uint8)


def overlay_mask(image_rgb, mask, color, alpha=0.45):
    colored = np.zeros_like(image_rgb)
    colored[mask.astype(bool)] = color
    return cv2.addWeighted(colored, alpha, image_rgb, 1 - alpha, 0)


def add_label(panel, text):
    labeled = np.full((PANEL_SIZE + LABEL_HEIGHT, PANEL_SIZE, 3), 255, dtype=np.uint8)
    labeled[LABEL_HEIGHT:, :, :] = panel
    cv2.putText(labeled, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return labeled


def build_row(image_path, mask_path, models_dict, device, img_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    _, val_tf = get_transforms(img_size)
    tensor = val_tf(image=image)["image"]

    panels = [add_label(image, "image"), add_label(overlay_mask(image, mask, (0, 200, 0)), "ground truth")]
    for name, model_or_list in models_dict.items():
        pred = predict_ensemble(model_or_list, tensor, device) if isinstance(model_or_list, list) \
            else predict(model_or_list, tensor, device)
        panels.append(add_label(overlay_mask(image, pred, (220, 30, 30)), name))

    row = np.concatenate(panels, axis=1)
    return cv2.cvtColor(row, cv2.COLOR_RGB2BGR)


def pick_non_empty(mask_paths, num_examples, seed, min_pixels=500):
    non_empty = [i for i in range(len(mask_paths))
                 if (cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE) > 0).sum() > min_pixels]
    random.Random(seed).shuffle(non_empty)
    return non_empty[:num_examples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/work/cvcs2026/bleedsense/datasets")
    parser.add_argument("--ckpt-dir", default="/work/cvcs2026/bleedsense/results/checkpoints")
    parser.add_argument("--output-dir", default="/work/cvcs2026/bleedsense/results/qualitative")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--architecture", default="unet",
                         help="Architecture used for the single-model baseline/aggressive panels")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- HemoSet -> Rabbani direction ---
    rabbani_images, rabbani_masks = load_rabbani_data(os.path.join(args.data_dir, "rabbani"))
    (_, _), (_, _), (test_img, test_mask) = split_rabbani_data(rabbani_images, rabbani_masks, seed=SPLIT_SEED)
    chosen = pick_non_empty(test_mask, args.num_examples, args.seed)

    baseline = load_checkpoint(args.architecture, f"hemoset_{args.architecture}_fold0", args.encoder, device, args.ckpt_dir)
    aggressive = load_checkpoint(args.architecture, f"hemoset_{args.architecture}_fold0_augaggressive",
                                  args.encoder, device, args.ckpt_dir)
    ensemble = [load_checkpoint(arch, f"hemoset_{arch}_fold0_augaggressive", args.encoder, device, args.ckpt_dir)
                for arch in ARCHITECTURES]
    models_dict = {"baseline (source-only)": baseline, "aggressive augmentation": aggressive, "ensemble": ensemble}

    for i, idx in enumerate(chosen):
        row = build_row(test_img[idx], test_mask[idx], models_dict, device)
        out_path = os.path.join(args.output_dir, f"hemoset_to_rabbani_ex{i + 1}.png")
        cv2.imwrite(out_path, row)
        print(f"Saved {out_path}", flush=True)

    del baseline, aggressive, ensemble

    # --- Rabbani -> HemoSet direction ---
    hemoset_images, hemoset_masks, _, _ = load_hemoset_data(os.path.join(args.data_dir, "hemoset"))
    chosen = pick_non_empty(list(hemoset_masks), args.num_examples, args.seed)

    baseline = load_checkpoint(args.architecture, f"rabbani_{args.architecture}", args.encoder, device, args.ckpt_dir)
    aggressive = load_checkpoint(args.architecture, f"rabbani_{args.architecture}_augaggressive",
                                  args.encoder, device, args.ckpt_dir)
    ensemble = [load_checkpoint(arch, f"rabbani_{arch}_augaggressive", args.encoder, device, args.ckpt_dir)
                for arch in ARCHITECTURES]
    models_dict = {"baseline (source-only)": baseline, "aggressive augmentation": aggressive, "ensemble": ensemble}

    for i, idx in enumerate(chosen):
        row = build_row(hemoset_images[idx], hemoset_masks[idx], models_dict, device)
        out_path = os.path.join(args.output_dir, f"rabbani_to_hemoset_ex{i + 1}.png")
        cv2.imwrite(out_path, row)
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
