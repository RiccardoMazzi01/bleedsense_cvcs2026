import argparse
import json
import os
import statistics as stats

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import (
    MedicalBleedingDataset,
    get_hemoset_all_images,
    get_rabbani_train_pool,
    get_transforms,
    load_hemoset_data,
    load_rabbani_data,
    split_rabbani_data,
)
from metrics import MetricTracker
from models import get_model

SPLIT_SEED = 42
ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]
METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95"]


class UnlabeledImageDataset(Dataset):
    """Solo immagini (nessuna maschera): usato per ricalibrare le BatchNorm sul pool
    non annotato del dominio target, stesso principio dei pool di adaptation in Fase 3.
    """

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]


def recalibrate_batchnorm(model, loader, device, max_batches=50):
    """AdaBN (Li et al., 2016): ricalcola media/varianza delle BatchNorm dell'encoder
    con forward pass (nessun backward, nessuna etichetta) su immagini non annotate del
    dominio target, prima della valutazione cross-dataset. Nessun peso aggiornato, solo
    le statistiche di normalizzazione; costo di un'inferenza extra su poche decine di batch.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # media cumulativa esatta sul pool, non EMA

    model.train()  # attiva l'aggiornamento delle statistiche BN nel forward
    with torch.no_grad():
        for i, images in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            model(images.to(device))
    model.eval()


def evaluate(model, loader, device):
    tracker = MetricTracker()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            tracker.update(model(images), masks)
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
                         help="Condizione i cui checkpoint vengono ricalibrati (default: la migliore, Fase 2)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--calib-batches", type=int, default=50,
                         help="Numero massimo di batch del pool target usati per ricalibrare le BN")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    _, val_tf = get_transforms(img_size)
    aug_suffix = f"_aug{args.augmentation}" if args.augmentation != "light" else ""

    # --- Dati condivisi (stessi split fissi di tutte le fasi) ---
    rabbani_images, rabbani_masks = load_rabbani_data(os.path.join(args.data_dir, "rabbani"))
    (_, _), (_, _), (rabbani_test_img, rabbani_test_mask) = split_rabbani_data(
        rabbani_images, rabbani_masks, seed=SPLIT_SEED
    )
    rabbani_test_ds = MedicalBleedingDataset(rabbani_test_img, rabbani_test_mask, transform=val_tf)
    rabbani_test_loader = DataLoader(rabbani_test_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    hemoset_images, hemoset_masks, _, _ = load_hemoset_data(os.path.join(args.data_dir, "hemoset"))
    hemoset_full_ds = MedicalBleedingDataset(hemoset_images, hemoset_masks, transform=val_tf)
    hemoset_full_loader = DataLoader(hemoset_full_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    rabbani_pool = get_rabbani_train_pool(args.data_dir, seed=SPLIT_SEED)
    rabbani_pool_loader = DataLoader(UnlabeledImageDataset(rabbani_pool, val_tf), batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)

    hemoset_pool = get_hemoset_all_images(args.data_dir)
    hemoset_pool_loader = DataLoader(UnlabeledImageDataset(hemoset_pool, val_tf), batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)

    rows = {"hemoset_to_rabbani": {"before": [], "after": []}, "rabbani_to_hemoset": {"before": [], "after": []}}

    # --- HemoSet -> Rabbani: ricalibra su pool Rabbani (train split, mai il test) ---
    for arch in ARCHITECTURES:
        for fold in range(5):
            run_name = f"hemoset_{arch}_fold{fold}{aug_suffix}"
            model = get_model(arch, encoder_name=args.encoder).to(device)
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{run_name}_best.pth"), map_location=device))
            model.eval()

            before = evaluate(model, rabbani_test_loader, device)
            recalibrate_batchnorm(model, rabbani_pool_loader, device, max_batches=args.calib_batches)
            after = evaluate(model, rabbani_test_loader, device)

            rows["hemoset_to_rabbani"]["before"].append(before)
            rows["hemoset_to_rabbani"]["after"].append(after)
            print(f"[{run_name}] Rabbani cross dice: {before['dice']:.4f} -> {after['dice']:.4f}", flush=True)

    # --- Rabbani -> HemoSet: ricalibra su pool HemoSet (tutto, transduttivo come in Fase 3) ---
    for arch in ARCHITECTURES:
        run_name = f"rabbani_{arch}{aug_suffix}"
        model = get_model(arch, encoder_name=args.encoder).to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{run_name}_best.pth"), map_location=device))
        model.eval()

        before = evaluate(model, hemoset_full_loader, device)
        recalibrate_batchnorm(model, hemoset_pool_loader, device, max_batches=args.calib_batches)
        after = evaluate(model, hemoset_full_loader, device)

        rows["rabbani_to_hemoset"]["before"].append(before)
        rows["rabbani_to_hemoset"]["after"].append(after)
        print(f"[{run_name}] HemoSet cross dice: {before['dice']:.4f} -> {after['dice']:.4f}", flush=True)

    # --- Riepilogo ---
    lines = ["| Direzione | Fase | " + " | ".join(m.upper() for m in METRICS) + " |",
             "|---|---" + "|---" * len(METRICS) + "|"]
    for label, key in [("HemoSet -> Rabbani (cross)", "hemoset_to_rabbani"),
                        ("Rabbani -> HemoSet (cross)", "rabbani_to_hemoset")]:
        for phase in ["before", "after"]:
            values = {m: [r[m] for r in rows[key][phase]] for m in METRICS}
            cells = " | ".join(fmt(values[m]) for m in METRICS)
            lines.append(f"| {label} | {'senza ricalibrazione' if phase == 'before' else 'con test-time BN recalibration'} | {cells} |")

    table_md = "\n".join(lines)
    print("\n" + table_md)

    out_path = os.path.join(args.output_dir, f"test_time_bn_{args.augmentation}_summary.md")
    with open(out_path, "w") as f:
        f.write(f"# Test-time BatchNorm recalibration (AdaBN) sopra augmentation={args.augmentation}\n\n")
        f.write(table_md + "\n")

    results_json_path = os.path.join(args.output_dir, f"test_time_bn_{args.augmentation}.json")
    with open(results_json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSalvato in {out_path} e {results_json_path}")


if __name__ == "__main__":
    main()
