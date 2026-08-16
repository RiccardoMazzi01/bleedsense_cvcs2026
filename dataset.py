import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalBleedingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # Binarizzazione (0 o 1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = np.expand_dims(mask, axis=0)

        return image, mask

def get_transforms(img_size=(256, 256)):
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


class AdaptedBleedingDataset(MedicalBleedingDataset):
    """Come MedicalBleedingDataset, ma prima del transform applica una funzione di
    appearance/domain adaptation (Reinhard, FDA, ...) usando un'immagine campionata
    a caso da un pool di immagini non annotate del dominio target (Fase 3).
    """

    def __init__(self, image_paths, mask_paths, target_image_paths, adapt_fn, transform=None, work_size=(256, 256)):
        super().__init__(image_paths, mask_paths, transform=transform)
        self.target_image_paths = target_image_paths
        self.adapt_fn = adapt_fn
        self.work_size = work_size  # ridimensiona prima dell'adaptation: FDA fa una FFT per canale,
        # farla a piena risoluzione sarebbe inutilmente lento dato che il modello lavora a work_size

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.work_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        target_path = random.choice(self.target_image_paths)
        target_image = cv2.imread(target_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.resize(target_image, self.work_size)
        image = self.adapt_fn(image, target_image)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = np.expand_dims(mask, axis=0)

        return image, mask


def load_hemoset_data(hemoset_dir):
    image_paths = []
    mask_paths = []
    groups = []  # ID maiale (es. pig1, pig2...)
    has_bleeding = []

    pig_folders = sorted(glob.glob(os.path.join(hemoset_dir, "pig*")))
    for pig_path in pig_folders:
        pig_id = os.path.basename(pig_path)

        img_dir = os.path.join(pig_path, "images")
        mask_dir = os.path.join(pig_path, "masks")

        imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        for img_p in imgs:
            fname = os.path.basename(img_p)

            # Gestione suffisso _mask.png
            ext = os.path.splitext(fname)[1]
            mask_fname = fname.replace(ext, f"_mask{ext}")
            mask_p = os.path.join(mask_dir, mask_fname)

            # Fallback se la maschera ha lo stesso identico nome
            if not os.path.exists(mask_p):
                mask_p = os.path.join(mask_dir, fname)

            if os.path.exists(mask_p):
                image_paths.append(img_p)
                mask_paths.append(mask_p)
                groups.append(pig_id)

                m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                has_bleeding.append(1 if (m is not None and (m > 0).any()) else 0)

    return np.array(image_paths), np.array(mask_paths), np.array(groups), np.array(has_bleeding)

def load_rabbani_data(rabbani_dir):
    img_dir = os.path.join(rabbani_dir, "images")
    mask_dir = os.path.join(rabbani_dir, "masks")

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))

    return np.array(image_paths), np.array(mask_paths)


def split_rabbani_data(image_paths, mask_paths, val_size=0.15, test_size=0.15, seed=42):
    """Split fisso train/val/test per Rabbani (nessun gruppo/soggetto disponibile nei metadati).

    Lo stesso seed va usato in tutte le fasi del progetto, cosi' il test set di Rabbani
    resta identico ovunque (baseline, augmentation, adaptation, joint training).
    """
    idx = np.arange(len(image_paths))
    train_idx, temp_idx = train_test_split(idx, test_size=(val_size + test_size), random_state=seed)
    rel_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test_size, random_state=seed)

    def subset(indices):
        return image_paths[indices], mask_paths[indices]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def get_aggressive_transforms(img_size=(256, 256)):
    """Augmentation 'aggressive' per la Fase 2 (Su et al., AAAI 2023 / setup chirurgico:
    luminosita'/contrasto, hue/saturation, gamma, blur, noise, compressione).

    Il val_transform e' identico a quello di get_transforms(), cosi' la valutazione resta
    comparabile tra gli esperimenti 'light' (Fase 1) e 'aggressive' (Fase 2): cambia solo
    cosa vede il modello in training, non come viene misurato.
    """
    train_transform = A.Compose([
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
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def get_rabbani_train_pool(data_dir, seed=42, val_size=0.15, test_size=0.15):
    """Solo le immagini (senza maschere) dello split di training di Rabbani: pool di
    immagini non annotate del dominio target per la Fase 3 (mai il val/test, per non
    contaminare la valutazione cross-dataset).
    """
    images, masks = load_rabbani_data(os.path.join(data_dir, "rabbani"))
    (train_img, _), _, _ = split_rabbani_data(images, masks, seed=seed, val_size=val_size, test_size=test_size)
    return train_img


def get_hemoset_all_images(data_dir):
    """Tutte le immagini di HemoSet (senza maschere): pool di immagini non annotate del
    dominio target per la Fase 3 quando si allena su Rabbani. Coerente con il fatto che
    tutto HemoSet e' gia' usato, senza etichette, come target di valutazione cross-dataset
    per i modelli allenati su Rabbani (setting transduttivo, esplicitamente sanzionato
    dalla tutor: 'puoi utilizzare anche immagini non annotate del dominio target').
    """
    images, _, _, _ = load_hemoset_data(os.path.join(data_dir, "hemoset"))
    return images
