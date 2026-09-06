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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

from dataset import (
    AdaptedBleedingDataset,
    MedicalBleedingDataset,
    get_aggressive_transforms,
    get_hemoset_all_images,
    get_rabbani_train_pool,
    get_transforms,
    load_hemoset_data,
    load_rabbani_data,
    split_rabbani_data,
)
from domain_adaptation import fda_transfer, reinhard_color_transfer
from metrics import MetricTracker
from models import get_model

AUGMENTATION_TRANSFORMS = {
    "light": get_transforms,
    "aggressive": get_aggressive_transforms,
}

ADAPTATION_FUNCTIONS = {
    "reinhard": reinhard_color_transfer,
    "fda": fda_transfer,
}

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


def build_hemoset_fold(data_dir, fold, img_size, augmentation="light", adaptation="none", use_blood_index=False):
    images, masks, groups, labels = load_hemoset_data(os.path.join(data_dir, "hemoset"))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    splits = list(sgkf.split(images, labels, groups))
    train_idx, val_idx = splits[fold]

    train_tf, _ = AUGMENTATION_TRANSFORMS[augmentation](img_size, use_blood_index=use_blood_index)
    _, val_tf = get_transforms(img_size, use_blood_index=use_blood_index)  # valutazione sempre senza augmentation, per confrontabilita'

    if adaptation == "none":
        train_ds = MedicalBleedingDataset(images[train_idx], masks[train_idx], transform=train_tf,
                                           use_blood_index=use_blood_index)
    else:
        target_pool = get_rabbani_train_pool(data_dir, seed=SPLIT_SEED)
        train_ds = AdaptedBleedingDataset(
            images[train_idx], masks[train_idx], target_pool,
            ADAPTATION_FUNCTIONS[adaptation], transform=train_tf, work_size=img_size,
        )

    val_ds = MedicalBleedingDataset(images[val_idx], masks[val_idx], transform=val_tf,
                                     use_blood_index=use_blood_index)
    return train_ds, val_ds


def build_hemoset_full(data_dir, img_size, use_blood_index=False):
    images, masks, _, _ = load_hemoset_data(os.path.join(data_dir, "hemoset"))
    _, val_tf = get_transforms(img_size, use_blood_index=use_blood_index)
    return MedicalBleedingDataset(images, masks, transform=val_tf, use_blood_index=use_blood_index)


def build_rabbani_splits(data_dir, img_size, augmentation="light", adaptation="none", use_blood_index=False):
    images, masks = load_rabbani_data(os.path.join(data_dir, "rabbani"))
    (train_img, train_mask), (val_img, val_mask), (test_img, test_mask) = split_rabbani_data(
        images, masks, seed=SPLIT_SEED
    )
    train_tf, _ = AUGMENTATION_TRANSFORMS[augmentation](img_size, use_blood_index=use_blood_index)
    _, val_tf = get_transforms(img_size, use_blood_index=use_blood_index)  # valutazione sempre senza augmentation, per confrontabilita'

    if adaptation == "none":
        train_ds = MedicalBleedingDataset(train_img, train_mask, transform=train_tf,
                                           use_blood_index=use_blood_index)
    else:
        target_pool = get_hemoset_all_images(data_dir)
        train_ds = AdaptedBleedingDataset(
            train_img, train_mask, target_pool,
            ADAPTATION_FUNCTIONS[adaptation], transform=train_tf, work_size=img_size,
        )

    val_ds = MedicalBleedingDataset(val_img, val_mask, transform=val_tf, use_blood_index=use_blood_index)
    test_ds = MedicalBleedingDataset(test_img, test_mask, transform=val_tf, use_blood_index=use_blood_index)
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
    parser.add_argument("--dataset", choices=["hemoset", "rabbani", "joint"], required=True)
    parser.add_argument("--architecture", choices=["unet", "unetplusplus", "deeplabv3plus"], required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--fold", type=int, default=0,
                         help="Indice fold (0-4), usato con --dataset hemoset o joint")
    parser.add_argument("--augmentation", choices=["light", "aggressive"], default="light",
                         help="Policy di data augmentation per il training (Fase 2)")
    parser.add_argument("--adaptation", choices=["none", "reinhard", "fda"], default="none",
                         help="Appearance/domain adaptation verso il dominio target (Fase 3)")
    parser.add_argument("--joint-sampling", choices=["natural", "balanced"], default="natural",
                         help="Solo con --dataset joint: 'balanced' pesca meta' batch da ciascun dominio "
                              "indipendentemente dalla sua dimensione (WeightedRandomSampler)")
    parser.add_argument("--use-blood-index", action="store_true",
                         help="Aggiunge un 4o canale di input R/(R+G+B), pensato per essere meno "
                              "sensibile a differenze di camera/illuminazione tra i due dataset. "
                              "Non ancora supportato insieme a --adaptation diversa da 'none'.")
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

    if args.use_blood_index and args.adaptation != "none":
        raise ValueError("--use-blood-index non e' ancora supportato insieme a --adaptation diversa da 'none'")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)

    run_name = f"{args.dataset}_{args.architecture}"
    run_name += f"_fold{args.fold}" if args.dataset in ("hemoset", "joint") else ""
    run_name += f"_aug{args.augmentation}" if args.augmentation != "light" else ""
    run_name += f"_adapt{args.adaptation}" if args.adaptation != "none" else ""
    run_name += "_balanced" if args.dataset == "joint" and args.joint_sampling == "balanced" else ""
    run_name += "_bloodindex" if args.use_blood_index else ""

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pth")

    # --- Dati ---
    is_joint = args.dataset == "joint"
    if args.dataset == "hemoset":
        train_ds, val_ds = build_hemoset_fold(args.data_dir, args.fold, img_size, args.augmentation, args.adaptation,
                                               use_blood_index=args.use_blood_index)
        cross_ds = build_rabbani_splits(args.data_dir, img_size, use_blood_index=args.use_blood_index)[2]  # test split di Rabbani
        cross_name = "rabbani_test"
        indomain_test_ds = val_ds  # per HemoSet il val set del fold e' il proxy in-domain
    elif args.dataset == "rabbani":
        train_ds, val_ds, indomain_test_ds = build_rabbani_splits(
            args.data_dir, img_size, args.augmentation, args.adaptation, use_blood_index=args.use_blood_index
        )
        cross_ds = build_hemoset_full(args.data_dir, img_size, use_blood_index=args.use_blood_index)  # tutto HemoSet, mai visto in training
        cross_name = "hemoset_full"
    else:
        # Fase 4: training congiunto su HemoSet (fold) + Rabbani (train split), valutazione
        # separata sui due test set held-out (nessun senso di "cross-dataset" qui: il modello
        # ha visto entrambi i domini in training).
        hemoset_train_ds, hemoset_eval_ds = build_hemoset_fold(
            args.data_dir, args.fold, img_size, args.augmentation, args.adaptation,
            use_blood_index=args.use_blood_index
        )
        rabbani_train_ds, _, rabbani_eval_ds = build_rabbani_splits(
            args.data_dir, img_size, args.augmentation, args.adaptation, use_blood_index=args.use_blood_index
        )
        train_ds = ConcatDataset([hemoset_train_ds, rabbani_train_ds])
        val_ds = hemoset_eval_ds  # segnale per early stopping/scheduler durante il training

    sampler = None
    if is_joint and args.joint_sampling == "balanced":
        # Peso per immagine inversamente proporzionale alla dimensione del proprio dominio:
        # ogni dominio ha ~50% di probabilita' di essere pescato in ciascun batch,
        # indipendentemente da quante immagini contribuisce (corregge lo sbilanciamento
        # osservato in Fase 4, dove HemoSet pesava ~59% solo perche' piu' numeroso).
        n_hemoset, n_rabbani = len(hemoset_train_ds), len(rabbani_train_ds)
        weights = [1.0 / n_hemoset] * n_hemoset + [1.0 / n_rabbani] * n_rabbani
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                               drop_last=True, sampler=sampler, shuffle=(sampler is None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if is_joint:
        hemoset_eval_loader = DataLoader(hemoset_eval_ds, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
        rabbani_eval_loader = DataLoader(rabbani_eval_ds, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
    else:
        indomain_test_loader = DataLoader(indomain_test_ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        cross_loader = DataLoader(cross_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Modello ---
    in_channels = 4 if args.use_blood_index else 3
    model = get_model(args.architecture, encoder_name=args.encoder, in_channels=in_channels).to(device)
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

    result = {
        "run_name": run_name,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "encoder": args.encoder,
        "augmentation": args.augmentation,
        "adaptation": args.adaptation,
        "joint_sampling": args.joint_sampling if args.dataset == "joint" else None,
        "use_blood_index": args.use_blood_index,
        "fold": args.fold if args.dataset in ("hemoset", "joint") else None,
        "seed": args.seed,
        "epochs_trained": len(history),
        "best_val_dice": best_dice,
        "history": history,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if is_joint:
        hemoset_tracker = MetricTracker()
        run_epoch(model, hemoset_eval_loader, device, tracker=hemoset_tracker)
        hemoset_metrics = hemoset_tracker.compute()

        rabbani_tracker = MetricTracker()
        run_epoch(model, rabbani_eval_loader, device, tracker=rabbani_tracker)
        rabbani_metrics = rabbani_tracker.compute()

        result["hemoset_test"] = hemoset_metrics
        result["rabbani_test"] = rabbani_metrics

        print(f"[{run_name}] FATTO. HemoSet dice={hemoset_metrics['dice']:.4f} "
              f"| Rabbani dice={rabbani_metrics['dice']:.4f}", flush=True)
    else:
        indomain_tracker = MetricTracker()
        run_epoch(model, indomain_test_loader, device, tracker=indomain_tracker)
        indomain_metrics = indomain_tracker.compute()

        cross_tracker = MetricTracker()
        run_epoch(model, cross_loader, device, tracker=cross_tracker)
        cross_metrics = cross_tracker.compute()

        result["indomain_test"] = indomain_metrics
        result["cross_dataset_test"] = {"target": cross_name, **cross_metrics}

        print(f"[{run_name}] FATTO. In-domain dice={indomain_metrics['dice']:.4f} "
              f"| Cross-dataset ({cross_name}) dice={cross_metrics['dice']:.4f}", flush=True)

    results_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{run_name}] Risultati salvati in {results_path}", flush=True)


if __name__ == "__main__":
    main()
