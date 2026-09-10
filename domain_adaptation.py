import cv2
import numpy as np


def reinhard_color_transfer(source_rgb, target_rgb):
    """Color transfer from Reinhard et al. (2001): matches the mean and standard
    deviation of the L*a*b* channels of the source image to those of a target
    image, leaving structure/content unchanged.
    """
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    src_mean, src_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1)) + 1e-6
    tgt_mean, tgt_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1)) + 1e-6

    result_lab = (source_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)


def fda_transfer(source_rgb, target_rgb, beta=0.01):
    """Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): replaces the low-frequency
    region of the source's amplitude spectrum with the target's (the 'style'), while
    keeping the source's phase (i.e. content/structure).

    beta controls the size of the swapped low-frequency region (fraction of
    height/width); small values (~0.01-0.05) are the ones used in the paper.
    """
    target_resized = cv2.resize(target_rgb, (source_rgb.shape[1], source_rgb.shape[0]))

    source = source_rgb.astype(np.float32)
    target = target_resized.astype(np.float32)

    result = np.zeros_like(source)
    h, w = source.shape[:2]
    b_h, b_w = max(1, int(h * beta)), max(1, int(w * beta))
    cy, cx = h // 2, w // 2

    for c in range(3):
        fft_src = np.fft.fftshift(np.fft.fft2(source[:, :, c]))
        fft_tgt = np.fft.fftshift(np.fft.fft2(target[:, :, c]))

        amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
        amp_tgt = np.abs(fft_tgt)

        amp_mixed = amp_src.copy()
        amp_mixed[cy - b_h:cy + b_h, cx - b_w:cx + b_w] = amp_tgt[cy - b_h:cy + b_h, cx - b_w:cx + b_w]

        fft_mixed = amp_mixed * np.exp(1j * pha_src)
        img_back = np.fft.ifft2(np.fft.ifftshift(fft_mixed))
        result[:, :, c] = np.real(img_back)

    return np.clip(result, 0, 255).astype(np.uint8)
