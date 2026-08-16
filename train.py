import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from dataset import (
    MedicalBleedingDataset,
    get_aggressive_transforms,
    get_transforms,
    load_hemoset_data,
    load_rabbani_data,
    split_rabbani_data,
)

AUGMENTATION_TRANSFORMS = {
    "light": get_transforms,
    "aggressive": get_aggressive_transforms,
}
from metrics import MetricTracker
from models import get_model

# Seed fisso per gli split dei dati: garantisce che ogni architettura/run veda
# esattamente gli stessi fold HemoSet e lo stesso train/val/test di Rabbani,
# indipendentemente dal seed usato per init pesi/augmentation (--seed).
SPLIT_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.eps) / (union + self.eps)).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


def build_hemoset_fold(data_dir, fold, img_size, augmentation="light"):
    images, masks, groups, labels = load_hemoset_data(os.path.join(data_dir, "hemoset"))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    splits = list(sgkf.split(images, labels, groups))
    train_idx, val_idx = splits[fold]

    train_tf, _ = AUGMENTATION_TRANSFORMS[augmentation](img_size)
    _, val_tf = get_transforms(img_size)  # valutazione sempre senza augmentation, per confrontabilita'
    train_ds = MedicalBleedingDataset(images[train_idx], masks[train_idx], transform=train_tf)
    val_ds = MedicalBleedingDataset(images[val_idx], masks[val_idx], transform=val_tf)
    return train_ds, val_ds


def build_hemoset_full(data_dir, img_size):
    images, masks, _, _ = load_hemoset_data(os.path.join(data_dir, "hemoset"))
    _, val_tf = get_transforms(img_size)
    return MedicalBleedingDataset(images, masks, transform=val_tf)


def build_rabbani_splits(data_dir, img_size, augmentation="light"):
    images, masks = load_rabbani_data(os.path.join(data_dir, "rabbani"))
    (train_img, train_mask), (val_img, val_mask), (test_img, test_mask) = split_rabbani_data(
        images, masks, seed=SPLIT_SEED
    )
    train_tf, _ = AUGMENTATION_TRANSFORMS[augmentation](img_size)
    _, val_tf = get_transforms(img_size)  # valutazione sempre senza augmentation, per confrontabilita'
    train_ds = MedicalBleedingDataset(train_img, train_mask, transform=train_tf)
    val_ds = MedicalBleedingDataset(val_img, val_mask, transform=val_tf)
    test_ds = MedicalBleedingDataset(test_img, test_mask, transform=val_tf)
    return train_ds, val_ds, test_ds


def run_epoch(model, loader, device, criterion=None, optimizer=None, tracker=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, masks)
                total_loss += loss.item() * images.size(0)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if tracker is not None:
                tracker.update(logits, masks)

    avg_loss = total_loss / len(loader.dataset) if criterion is not None else None
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hemoset", "rabbani"], required=True)
    parser.add_argument("--architecture", choices=["unet", "unetplusplus", "deeplabv3plus"], required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--fold", type=int, default=0, help="Indice fold (0-4), usato solo con --dataset hemoset")
    parser.add_argument("--augmentation", choices=["light", "aggressive"], default="light",
                         help="Policy di data augmentation per il training (Fase 2)")
    parser.add_argument("--data-dir", default="/work/cvcs2026/bleedsense/datasets")
    parser.add_argument("--output-dir", default="/work/cvcs2026/bleedsense/results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Seed per init pesi/augmentation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping su Dice di validazione")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)

    run_name = f"{args.dataset}_{args.architecture}"
    run_name += f"_fold{args.fold}" if args.dataset == "hemoset" else ""
    run_name += f"_aug{args.augmentation}" if args.augmentation != "light" else ""

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pth")

    # --- Dati ---
    if args.dataset == "hemoset":
        train_ds, val_ds = build_hemoset_fold(args.data_dir, args.fold, img_size, args.augmentation)
        cross_ds = build_rabbani_splits(args.data_dir, img_size)[2]  # test split di Rabbani
        cross_name = "rabbani_test"
        indomain_test_ds = val_ds  # per HemoSet il val set del fold e' il proxy in-domain
    else:
        train_ds, val_ds, indomain_test_ds = build_rabbani_splits(args.data_dir, img_size, args.augmentation)
        cross_ds = build_hemoset_full(args.data_dir, img_size)  # tutto HemoSet, mai visto in training
        cross_name = "hemoset_full"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    indomain_test_loader = DataLoader(indomain_test_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    cross_loader = DataLoader(cross_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Modello ---
    model = get_model(args.architecture, encoder_name=args.encoder).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_dice = -1.0
    epochs_no_improve = 0
    history = []

    print(f"=== Run: {run_name} | device={device} | train={len(train_ds)} val={len(val_ds)} ===", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, criterion=criterion, optimizer=optimizer)

        val_tracker = MetricTracker()
        val_loss = run_epoch(model, val_loader, device, criterion=criterion, tracker=val_tracker)
        val_metrics = val_tracker.compute()
        scheduler.step(val_metrics["dice"])

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        print(f"[{run_name}] epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[{run_name}] early stopping all'epoca {epoch} "
                      f"(nessun miglioramento da {args.patience} epoche)", flush=True)
                break

    # --- Valutazione finale con il best checkpoint ---
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    indomain_tracker = MetricTracker()
    run_epoch(model, indomain_test_loader, device, tracker=indomain_tracker)
    indomain_metrics = indomain_tracker.compute()

    cross_tracker = MetricTracker()
    run_epoch(model, cross_loader, device, tracker=cross_tracker)
    cross_metrics = cross_tracker.compute()

    result = {
        "run_name": run_name,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "encoder": args.encoder,
        "augmentation": args.augmentation,
        "fold": args.fold if args.dataset == "hemoset" else None,
        "seed": args.seed,
        "epochs_trained": len(history),
        "best_val_dice": best_dice,
        "indomain_test": indomain_metrics,
        "cross_dataset_test": {"target": cross_name, **cross_metrics},
        "history": history,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    results_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{run_name}] FATTO. In-domain dice={indomain_metrics['dice']:.4f} "
          f"| Cross-dataset ({cross_name}) dice={cross_metrics['dice']:.4f}", flush=True)
    print(f"[{run_name}] Risultati salvati in {results_path}", flush=True)


if __name__ == "__main__":
    main()
