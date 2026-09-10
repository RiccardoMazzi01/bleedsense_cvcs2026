import argparse
import json
import os
import statistics as stats

import torch
from torch.utils.data import DataLoader

from dataset import (
    MedicalBleedingDataset,
    get_transforms,
    load_hemoset_data,
    load_rabbani_data,
    split_rabbani_data,
)
from metrics import MetricTracker
from models import get_model
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_SEED = 42
ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]
METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95"]


def load_ensemble(run_names, encoder, device, ckpt_dir):
    models = []
    for run_name in run_names:
        model = get_model(run_names[run_name], encoder_name=encoder).to(device)
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
    return models


def ensemble_evaluate(models, loader, device):
    tracker = MetricTracker()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            probs_sum = None
            for model in models:
                probs = torch.sigmoid(model(images))
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs_avg = probs_sum / len(models)
            tracker.update(probs_avg, masks, from_logits=False)
    return tracker.compute()


def mean_std(values):
    m = stats.mean(values)
    s = stats.stdev(values) if len(values) > 1 else 0.0
    return m, s


def fmt(values):
    m, s = mean_std(values)
    return f"{m:.3f}±{s:.3f}" if len(values) > 1 else f"{m:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/work/cvcs2026/bleedsense/datasets")
    parser.add_argument("--output-dir", default="/work/cvcs2026/bleedsense/results")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--augmentation", default="aggressive",
                         help="Condition whose checkpoints get ensembled (default: the best one, Phase 2)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    _, val_tf = get_transforms(img_size)

    aug_suffix = f"_aug{args.augmentation}" if args.augmentation != "light" else ""

    # --- Shared data ---
    rabbani_images, rabbani_masks = load_rabbani_data(os.path.join(args.data_dir, "rabbani"))
    (_, _), (_, _), (rabbani_test_img, rabbani_test_mask) = split_rabbani_data(
        rabbani_images, rabbani_masks, seed=SPLIT_SEED
    )
    rabbani_test_ds = MedicalBleedingDataset(rabbani_test_img, rabbani_test_mask, transform=val_tf)
    rabbani_test_loader = DataLoader(rabbani_test_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    hemoset_images, hemoset_masks, hemoset_groups, hemoset_labels = load_hemoset_data(
        os.path.join(args.data_dir, "hemoset")
    )
    hemoset_full_ds = MedicalBleedingDataset(hemoset_images, hemoset_masks, transform=val_tf)
    hemoset_full_loader = DataLoader(hemoset_full_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    results = {"augmentation": args.augmentation, "hemoset_indomain_per_fold": [], "rabbani_cross_per_fold": []}

    # --- Ensemble of the 3 architectures, for each HemoSet fold ---
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    splits = list(sgkf.split(hemoset_images, hemoset_labels, hemoset_groups))

    for fold, (_, val_idx) in enumerate(splits):
        run_names = {f"hemoset_{arch}_fold{fold}{aug_suffix}": arch for arch in ARCHITECTURES}
        models = load_ensemble(run_names, args.encoder, device, ckpt_dir)

        hemoset_val_ds = MedicalBleedingDataset(hemoset_images[val_idx], hemoset_masks[val_idx], transform=val_tf)
        hemoset_val_loader = DataLoader(hemoset_val_ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)

        indomain_metrics = ensemble_evaluate(models, hemoset_val_loader, device)
        cross_metrics = ensemble_evaluate(models, rabbani_test_loader, device)

        results["hemoset_indomain_per_fold"].append(indomain_metrics)
        results["rabbani_cross_per_fold"].append(cross_metrics)

        print(f"[fold {fold}] HemoSet in-domain dice={indomain_metrics['dice']:.4f} "
              f"| Rabbani cross dice={cross_metrics['dice']:.4f}", flush=True)

    # --- Ensemble of the 3 architectures, Rabbani run (single, no folds) ---
    run_names = {f"rabbani_{arch}{aug_suffix}": arch for arch in ARCHITECTURES}
    models = load_ensemble(run_names, args.encoder, device, ckpt_dir)
    results["rabbani_indomain"] = ensemble_evaluate(models, rabbani_test_loader, device)
    results["hemoset_cross"] = ensemble_evaluate(models, hemoset_full_loader, device)
    print(f"[rabbani] Rabbani in-domain dice={results['rabbani_indomain']['dice']:.4f} "
          f"| HemoSet cross dice={results['hemoset_cross']['dice']:.4f}", flush=True)

    # --- Summary ---
    lines = ["| Setting | DICE | IOU | PRECISION | RECALL | F1 | HD95 |", "|---|---|---|---|---|---|---|"]

    for label, per_fold in [("HemoSet -> HemoSet (in-domain, ensemble)", results["hemoset_indomain_per_fold"]),
                             ("HemoSet -> Rabbani (cross, ensemble)", results["rabbani_cross_per_fold"])]:
        values = {m: [r[m] for r in per_fold] for m in METRICS}
        cells = " | ".join(fmt(values[m]) for m in METRICS)
        lines.append(f"| {label} | {cells} |")

    for label, single in [("Rabbani -> Rabbani (in-domain, ensemble)", results["rabbani_indomain"]),
                           ("Rabbani -> HemoSet (cross, ensemble)", results["hemoset_cross"])]:
        cells = " | ".join(f"{single[m]:.3f}" for m in METRICS)
        lines.append(f"| {label} | {cells} |")

    table_md = "\n".join(lines)
    print("\n" + table_md)

    out_path = os.path.join(args.output_dir, f"ensemble_{args.augmentation}_summary.md")
    with open(out_path, "w") as f:
        f.write(f"# Ensemble of the 3 architectures (augmentation={args.augmentation})\n\n")
        f.write(table_md + "\n")

    results_json_path = os.path.join(args.output_dir, f"ensemble_{args.augmentation}.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path} and {results_json_path}")


if __name__ == "__main__":
    main()
