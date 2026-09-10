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

def compute_blood_index(image_rgb):
    """Extra channel derived from RGB, meant to be less sensitive to camera/illumination
    differences between HemoSet and Rabbani than the raw RGB channels: ratio R/(R+G+B)
    (excess-red-like index, in [0, 1]), used in the literature to highlight hemorrhagic
    regions in a way that is relatively invariant to exposure.
    """
    img = image_rgb.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    index = r / (r + g + b + 1e-6)
    return index.astype(np.float32)


class MedicalBleedingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, use_blood_index=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.use_blood_index = use_blood_index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # binarize (0 or 1)

        blood_index = compute_blood_index(image) if self.use_blood_index else None

        if self.transform is not None:
            if blood_index is not None:
                augmented = self.transform(image=image, mask=mask, blood_index=blood_index)
                blood_index = augmented['blood_index']
            else:
                augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = np.expand_dims(mask, axis=0)

        if blood_index is not None:
            if not isinstance(blood_index, torch.Tensor):
                blood_index = torch.from_numpy(blood_index)
            blood_index = blood_index.to(image.dtype)
            # rescaled from [0, 1] to roughly [-2, 2], the same order of magnitude as the
            # RGB channels normalized above with the ImageNet statistics
            blood_index = (blood_index - 0.5) * 4.0
            if blood_index.ndim == 2:
                blood_index = blood_index.unsqueeze(0)
            image = torch.cat([image, blood_index], dim=0)

        return image, mask

def get_transforms(img_size=(256, 256), use_blood_index=False):
    # blood_index is registered as a target of type 'mask': it only receives the
    # geometric transforms (resize/flip), not the color ones, since it's not an RGB channel.
    additional_targets = {"blood_index": "mask"} if use_blood_index else None

    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets=additional_targets)

    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets=additional_targets)

    return train_transform, val_transform


class AdaptedBleedingDataset(MedicalBleedingDataset):
    """Like MedicalBleedingDataset, but before the transform it applies an
    appearance/domain adaptation function (Reinhard, FDA, ...) using an image sampled
    at random from a pool of unlabeled target-domain images (Phase 3).
    """

    def __init__(self, image_paths, mask_paths, target_image_paths, adapt_fn, transform=None, work_size=(256, 256)):
        super().__init__(image_paths, mask_paths, transform=transform)
        self.target_image_paths = target_image_paths
        self.adapt_fn = adapt_fn
        self.work_size = work_size  # resize before adaptation: FDA does a per-channel FFT,
        # doing it at full resolution would be needlessly slow since the model works at work_size anyway

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.work_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.work_size, interpolation=cv2.INTER_NEAREST)
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
    groups = []  # pig ID (e.g. pig1, pig2...)
    has_bleeding = []

    pig_folders = sorted(glob.glob(os.path.join(hemoset_dir, "pig*")))
    for pig_path in pig_folders:
        pig_id = os.path.basename(pig_path)

        img_dir = os.path.join(pig_path, "images")
        mask_dir = os.path.join(pig_path, "masks")

        imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        for img_p in imgs:
            fname = os.path.basename(img_p)

            # Handle the _mask.png suffix
            ext = os.path.splitext(fname)[1]
            mask_fname = fname.replace(ext, f"_mask{ext}")
            mask_p = os.path.join(mask_dir, mask_fname)

            # Fallback if the mask has the exact same name
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
    """Fixed train/val/test split for Rabbani (no group/subject available in the metadata).

    The same seed must be used across all phases of the project, so the Rabbani test set
    stays identical everywhere (baseline, augmentation, adaptation, joint training).
    """
    idx = np.arange(len(image_paths))
    train_idx, temp_idx = train_test_split(idx, test_size=(val_size + test_size), random_state=seed)
    rel_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test_size, random_state=seed)

    def subset(indices):
        return image_paths[indices], mask_paths[indices]

    return subset(train_idx), subset(val_idx), subset(test_idx)


def get_aggressive_transforms(img_size=(256, 256), use_blood_index=False):
    """'Aggressive' augmentation for Phase 2 (Su et al., AAAI 2023 / surgical setting:
    brightness/contrast, hue/saturation, gamma, blur, noise, compression).

    val_transform is identical to the one in get_transforms(), so evaluation stays
    comparable between the 'light' (Phase 1) and 'aggressive' (Phase 2) experiments: only
    what the model sees during training changes, not how it is measured.
    """
    additional_targets = {"blood_index": "mask"} if use_blood_index else None

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
    ], additional_targets=additional_targets)

    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets=additional_targets)

    return train_transform, val_transform


def get_rabbani_train_pool(data_dir, seed=42, val_size=0.15, test_size=0.15):
    """Only the images (no masks) of the Rabbani training split: a pool of unlabeled
    target-domain images for Phase 3 (never val/test, to avoid contaminating the
    cross-dataset evaluation).
    """
    images, masks = load_rabbani_data(os.path.join(data_dir, "rabbani"))
    (train_img, _), _, _ = split_rabbani_data(images, masks, seed=seed, val_size=val_size, test_size=test_size)
    return train_img


def get_hemoset_all_images(data_dir):
    """All HemoSet images (no masks): a pool of unlabeled target-domain images for
    Phase 3 when training on Rabbani. Consistent with the fact that all of HemoSet is
    already used, unlabeled, as the cross-dataset evaluation target for models trained
    on Rabbani (a transductive setting, explicitly sanctioned: 'you can also use
    unlabeled images from the target domain').
    """
    images, _, _, _ = load_hemoset_data(os.path.join(data_dir, "hemoset"))
    return images
