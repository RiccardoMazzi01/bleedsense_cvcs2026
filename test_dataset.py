import os
from dataset import load_hemoset_data, load_rabbani_data, MedicalBleedingDataset, get_transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = "/work/cvcs2026/bleedsense/datasets"

print("--- Test Caricamento HemoSet ---")
images, masks, groups, labels = load_hemoset_data(os.path.join(BASE_DIR, "hemoset"))
print(f"HemoSet Trovati: {len(images)} coppie immagine-maschera across {len(set(groups))} maiali.")

sgkf = StratifiedGroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(images, labels, groups)):
    val_pigs = set(groups[val_idx])
    train_pigs = set(groups[train_idx])
    print(f"Fold {fold}: Train pigs={len(train_pigs)}, Val pigs={len(val_pigs)} -> {val_pigs}")

print("\n--- Test Caricamento Rabbani ---")
r_images, r_masks = load_rabbani_data(os.path.join(BASE_DIR, "rabbani"))
print(f"Rabbani Trovati: {len(r_images)} immagini, {len(r_masks)} maschere.")

print("\n--- Test PyTorch DataLoader ---")
train_tf, _ = get_transforms()
dataset = MedicalBleedingDataset(images[:10], masks[:10], transform=train_tf)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

img_b, mask_b = next(iter(loader))
print(f"Batch Immagini: {img_b.shape} (dtype: {img_b.dtype})")
print(f"Batch Maschere: {mask_b.shape} (dtype: {mask_b.dtype})")
print("Test completato con successo!")
