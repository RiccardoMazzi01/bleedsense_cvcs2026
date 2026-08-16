import cv2
import numpy as np


def reinhard_color_transfer(source_rgb, target_rgb):
    """Color transfer di Reinhard et al. (2001): abbina media e deviazione standard
    dei canali L*a*b* dell'immagine sorgente a quelli di un'immagine target,
    lasciando invariata la struttura/contenuto.
    """
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    src_mean, src_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1)) + 1e-6
    tgt_mean, tgt_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1)) + 1e-6

    result_lab = (source_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)


def fda_transfer(source_rgb, target_rgb, beta=0.01):
    """Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): sostituisce la regione
    a bassa frequenza dello spettro di ampiezza della sorgente con quella del target
    (lo 'stile'), mantenendo la fase della sorgente (quindi il contenuto/struttura).

    beta controlla la dimensione della regione a bassa frequenza scambiata (frazione
    di altezza/larghezza); valori piccoli (~0.01-0.05) sono quelli usati nel paper.
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
