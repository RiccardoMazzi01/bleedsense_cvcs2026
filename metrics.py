import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _to_numpy_binary(x, threshold=0.5, from_logits=False):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if from_logits:
        x = 1.0 / (1.0 + np.exp(-x))
    return (x > threshold).astype(np.uint8)


def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return float((2 * intersection + eps) / (union + eps))


def iou_score(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def precision_recall_f1(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


def _boundary(mask):
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    return mask & ~eroded


def hausdorff_distance_95(pred, target):
    """HD95 (in pixel) tra due maschere binarie 2D.

    Implementazione custom via distance transform (scipy), non e' garantita
    numericamente identica a librerie come medpy/MONAI: va bene per confronti
    interni al progetto, ma se serve confrontare direttamente coi valori HD95
    riportati nei paper originali va rivalidata con la stessa libreria che
    hanno usato loro.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        h, w = target.shape
        return float(np.sqrt(h ** 2 + w ** 2))

    pred_b = _boundary(pred)
    target_b = _boundary(target)

    dt_target = distance_transform_edt(~target_b)
    dt_pred = distance_transform_edt(~pred_b)

    d_pred_to_target = dt_target[pred_b]
    d_target_to_pred = dt_pred[target_b]

    all_d = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_d, 95))


class MetricTracker:
    """Accumula le metriche immagine per immagine su un intero DataLoader/epoca."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._dice, self._iou = [], []
        self._precision, self._recall, self._f1 = [], [], []
        self._hd95 = []

    def update(self, logits, target, from_logits=True):
        preds = _to_numpy_binary(logits, threshold=self.threshold, from_logits=from_logits)
        targets = _to_numpy_binary(target, threshold=0.5, from_logits=False)

        for p, t in zip(preds, targets):
            p2d, t2d = p[0], t[0]  # rimuove la dimensione canale (1, H, W) -> (H, W)
            self._dice.append(dice_coefficient(p2d, t2d))
            self._iou.append(iou_score(p2d, t2d))
            prec, rec, f1 = precision_recall_f1(p2d, t2d)
            self._precision.append(prec)
            self._recall.append(rec)
            self._f1.append(f1)
            self._hd95.append(hausdorff_distance_95(p2d, t2d))

    def compute(self):
        return {
            "dice": float(np.mean(self._dice)) if self._dice else 0.0,
            "iou": float(np.mean(self._iou)) if self._iou else 0.0,
            "precision": float(np.mean(self._precision)) if self._precision else 0.0,
            "recall": float(np.mean(self._recall)) if self._recall else 0.0,
            "f1": float(np.mean(self._f1)) if self._f1 else 0.0,
            "hd95": float(np.mean(self._hd95)) if self._hd95 else 0.0,
        }
