# BleedSense — Cross-Dataset Robustness for Surgical Blood Segmentation

**Final report — Computer Vision and Cognitive Systems (CVCS) course, A.Y. 2025/2026, UNIMORE**

Riccardo Mazzi (294437)
Supervisor: Dr. Livia Del Gaudio (AImageLab, TRAMIS project)

---

## Abstract

Automatic hemostasis management in robotic surgery requires reliable segmentation of blood in the operative field. Segmentation models trained on a single surgical dataset, however, generalize poorly to other datasets with different visual characteristics (lighting, tissue, instrumentation). This work studies the problem on two public datasets, **HemoSet** (robotic surgery on a porcine model) and **Rabbani et al.** (gynecological laparoscopy on humans), confirming an **asymmetric domain gap** already observed in previous work from the research group, and systematically compares six families of strategies to reduce it: aggressive data augmentation, appearance/domain adaptation (Reinhard color transfer, Fourier Domain Adaptation), multi-domain joint training (with and without domain balancing), ensembling of multiple architectures, an illumination-invariant input channel, and feature-level style mixing (MixStyle), plus a test-time BatchNorm recalibration. Across three standard segmentation architectures (UNet, UNet++, DeepLabV3+, `resnet34` encoder), the most effective strategy is **aggressive data augmentation**, which improves cross-dataset robustness in both directions with no appreciable in-domain cost; combined with an **ensemble of the three architectures** at inference time, it yields the best overall result. Three further, more original extensions attempted afterwards did not improve on this result any further, but their analysis provides useful insight into why more targeted interventions fail in this setting.

---

## 1. Introduction and Motivation

Automating hemostasis management in robotic surgery — identifying and controlling bleeding during the procedure — requires, as a first step, reliable segmentation of blood in the endoscopic field of view. This project is part of the **TRAMIS** research initiative, focused on real-time bleeding detection in robotic surgery; since real, annotated TRAMIS surgical data were not yet available at the time of this work, the study was conducted on two public reference datasets in the domain, used as proxies:

- **HemoSet** (Miao et al., ISMR 2024): induced bleeding during teleoperated robotic surgery on a porcine model, 962 image-mask pairs, 10 subjects.
- **Rabbani et al.** (MIDL 2022): gynecological laparoscopy on human patients, 751 annotated images.

Previous work from the research group (Giusti & Marzo, CVCS 2025/2026 project report, see Section 2) had observed an **asymmetric** performance drop when a model trained on one of the two datasets is evaluated on the other: **HemoSet → Rabbani** shows a strong general performance drop, while **Rabbani → HemoSet** shows a marked increase in false positives (non-hemorrhagic regions segmented as blood).

**Project goal**: to quantify this asymmetry rigorously and reproducibly, and to systematically study which strategies actually increase the cross-dataset robustness of segmentation models, comparing multiple experimental conditions without the constraint of converging on a single solution.

---

## 2. Related Work

- **Miao et al.**, "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management", ISMR 2024 — the HemoSet dataset; benchmark of standard segmentation models, IoU/F1/Hausdorff Distance metrics.
- **Rabbani, Seve, Bourdel & Bartoli**, "Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation", MIDL 2022 — dataset and segmentation system with an adversarial domain adaptation component; IoU and F-Score metrics.
- **Yang & Soatto**, "FDA: Fourier Domain Adaptation for Semantic Segmentation", CVPR 2020 — swapping low-frequency amplitude components in the Fourier spectrum, used here as an appearance adaptation method (Section 4.2).
- **Reinhard, Ashikhmin, Gooch & Shirley**, "Color Transfer between Images", IEEE CG&A 2001 — statistical color transfer in L\*a\*b\* space, the second appearance adaptation method used.
- **Zhou et al.**, "Domain Generalization with MixStyle", ICLR 2021 — feature-level style mixing, used as an extension (Section 5.2).
- **Li et al.**, "Revisiting Batch Normalization For Practical Domain Adaptation" (AdaBN), 2016 — test-time BatchNorm recalibration, used as an extension (Section 5.2).
- **Su et al.**, AAAI 2023 — single-source domain generalization via augmentation.
- **Giusti & Marzo**, CVCS 2025/2026 project report — first quantification of the asymmetric domain gap between HemoSet and Rabbani (zero-shot Dice: HemoSet→Rabbani 0.29, Rabbani→HemoSet 0.56), qualitatively consistent with Section 4.1 despite different splits/hyperparameters.

---

## 3. Approach

### 3.1 Datasets and Splits

- **HemoSet**: 962 image-mask pairs, 10 subjects (pigs). **Stratified Group K-Fold** split into 5 folds, with the subject as the group, to avoid the same animal appearing in both training and validation within the same fold.
- **Rabbani**: 751 images/masks. Fixed **70/15/15** split (train/val/test, seed 42), reused identically across all phases of the project.

### 3.2 Architectures and Training Protocol

Three standard architectures via `segmentation_models_pytorch`, same `resnet34` encoder for a fair comparison: **UNet**, **UNet++**, **DeepLabV3+**. Dice+BCE loss, AdamW optimizer (lr 1e-4), `ReduceLROnPlateau` scheduler, batch size 8, 256×256 images, the full dataset for each split. Early stopping on validation Dice (patience 8, cap of 40 epochs) — empirically verified across all 87 Phase 1-4 runs that this budget was sufficient for convergence (only 2/87 runs reached the cap, both with a negligible residual improvement, <0.01 Dice). The exact same configuration was used for every condition compared, to isolate the effect of the single variable under study.

### 3.3 Metrics and Evaluation Protocol

Dice, IoU, Precision, Recall, F1, **HD95** (95th-percentile Hausdorff Distance) — this set fully covers the metrics used in the original papers of both datasets, with HD95 preferred over the maximum HD because it is less sensitive to single-pixel outliers. Protocol: a model trained on HemoSet is evaluated in-domain on the validation fold and cross-dataset on the fixed Rabbani test set; a model trained on Rabbani is evaluated in-domain on its own fixed test set and cross-dataset on the whole of HemoSet (never seen during training).

---

## 4. Results

### 4.1 Baseline and Domain Gap Asymmetry (Phase 1)

The three architectures, trained separately on the two datasets (minimal augmentation), confirm and quantify the asymmetry observed in prior work: for **HemoSet→Rabbani**, mean Dice drops by about 70% relative (0.745 → 0.217), with HD95 rising from ~29px to ~130px, driven by a collapse in **recall** (0.76→0.19): the model becomes too conservative on the target domain, missing most of the actual blood. For **Rabbani→HemoSet**, the drop is more contained (0.684 → 0.585) but **precision** collapses (0.74→0.50) while recall stays high — consistent with the original observation about false positives. The three architectures behave very similarly under every condition: none resolves the domain gap on its own, motivating the shift in focus toward training/adaptation strategies rather than architecture choice.

### 4.2 Data Augmentation, Appearance Adaptation, Joint Training (Phases 2-4)

An "aggressive" augmentation (hue/saturation, gamma, blur, noise, JPEG compression, via Albumentations) **improves cross-dataset robustness across all three architectures in both directions**, at almost no in-domain cost (HemoSet→Rabbani: 0.217→0.321, +48% relative; Rabbani→HemoSet: 0.585→0.601). Two appearance adaptation methods — Reinhard color transfer (2001) and Fourier Domain Adaptation (Yang & Soatto, CVPR 2020), applied using unlabeled target-domain images — help the more critical direction (HemoSet→Rabbani: 0.217→0.263 with Reinhard, →0.242 with FDA) but **at a non-negligible in-domain cost** (e.g., Rabbani in-domain: 0.684→0.611 with Reinhard) and do not beat aggressive augmentation. The hypothesis is that, unlike augmentation (probabilistic, applied with some probability per image), adaptation here is **always active** during training: the model never sees the "clean" appearance of its own source domain, and ends up split between two styles instead of properly consolidating one with variations.

Training on the union of HemoSet and Rabbani (**joint training**, with separate held-out test sets) was not a solution: no systematic benefit on HemoSet, and a systematic drop on Rabbani (0.684→0.636) caused by the size imbalance between the two datasets in the combined training set (~59% HemoSet vs ~41% Rabbani). A domain-balanced sampling scheme (`WeightedRandomSampler`, ~50/50 per batch) recovers only about half of the drop (0.636→0.656), without making joint training competitive with the other strategies.

### 4.3 Ensemble of the Three Architectures

As an original contribution beyond the assigned roadmap, the three architectures' predictions were ensembled — at zero additional training cost (simply averaging the sigmoid probabilities of the already-trained aggressive-augmentation checkpoints) — the result beats the best single model on **all four** tested conditions, with modest but systematic, exception-free gains (e.g., HemoSet→Rabbani: 0.337 best single model → 0.343 ensemble; Rabbani→HemoSet: 0.615 → 0.623).

### 4.4 Comparative Summary

Mean over the three architectures for each strategy (full per-architecture breakdown in the supplementary materials):

| Strategy | HemoSet (in-domain) | Rabbani (in-domain) | HemoSet→Rabbani (cross) | Rabbani→HemoSet (cross) |
|---|---|---|---|---|
| Baseline (source-only) | 0.745 | 0.684 | 0.217 | 0.585 |
| Aggressive augmentation | 0.747 | 0.681 | 0.321 | 0.601 |
| Appearance adaptation (Reinhard) | 0.719 | 0.611 | 0.263 | 0.577 |
| Appearance adaptation (FDA) | 0.720 | 0.675 | 0.242 | 0.550 |
| Natural joint training | 0.742 | 0.636 | n/a* | n/a* |
| Balanced joint training | 0.739 | 0.656 | n/a* | n/a* |
| **Ensemble of 3 architectures (on aggressive)** | **0.770** | **0.706** | **0.343** | **0.623** |

\* joint training is evaluated in-domain on both test sets, so there is no comparable "cross-dataset" condition for this row.

![Dice comparison by strategy and evaluation condition](chart placeholder — see HTML/PDF version of the report)

**Recommended configuration: aggressive augmentation in training + ensemble of the three architectures at inference** — the only combination that improves cross-dataset robustness in both directions without penalizing in-domain performance.

---

## 5. Discussion

### 5.1 Why Augmentation Beats Adaptation and Joint Training

The common factor behind the two less effective strategies — always-active appearance adaptation, imbalanced joint training — appears to be the lack of a mechanism that still preserves the model's exposure to the "clean" distribution of its own source domain during training. A **light, probabilistic** intervention on visual variety (augmentation) beats more targeted but **always-active** interventions on style, or simply mixing the data together.

### 5.2 Three Further Extensions (Negative but Informative Outcome)

After identifying the recommended configuration, three additional original extensions were attempted, to check whether it could be improved further:

1. **"Blood-index" input channel** (ratio R/(R+G+B), in theory less sensitive to camera/illumination differences): slightly improves HemoSet→Rabbani (0.321→0.331) but **markedly worsens** Rabbani→HemoSet (0.601→0.526). The index, computed on raw pixels, carries over the white-balance differences between the datasets instead of ignoring them, becoming a shortcut feature specific to the source domain rather than a transferable signal.
2. **MixStyle** (Zhou et al., ICLR 2021): mixes the encoder's feature-map statistics (mean/variance) between samples in the same batch, without requiring target-domain images. Result: a diffuse but small degradation in both cross-dataset directions (0.217→0.203; 0.585→0.561), likely because it is applied too early in the encoder (where edge/texture information relevant to precise contours is encoded) and because training batches remain single-domain in this setting, limiting the actual style diversity injected.
3. **Test-time BatchNorm recalibration** (AdaBN, Li et al. 2016): recalibrates BatchNorm statistics on unlabeled target-domain images at inference time, with no additional training. Shifts the precision/recall trade-off in opposite directions in the two cross-dataset directions (HemoSet→Rabbani: precision 0.382→0.283, recall 0.391→0.546; Rabbani→HemoSet: the exact opposite) but net Dice does not improve in either direction (0.321→0.318; 0.601→0.570).

| Extension | HemoSet→Rabbani (cross) | Rabbani→HemoSet (cross) | Outcome |
|---|---|---|---|
| Blood-index channel | 0.321→0.331 | 0.601→0.526 | Negative (unfavorable trade-off) |
| MixStyle | 0.217→0.203 | 0.585→0.561 | Negative (no gain) |
| Test-time BN recalibration | 0.321→0.318 | 0.601→0.570 | Neutral/negative |

None of the three beats the recommended configuration. The pattern is nonetheless informative: three very different mechanisms fail in the same way — either introducing a shortcut specific to the source domain, or disturbing useful information without replacing it with something more transferable — reinforcing the conclusion that aggressive augmentation + ensemble is not a local optimum that can be easily improved with light interventions, at least for this pair of datasets.

---

## 6. Limitations and Future Work

- Applying appearance/domain adaptation with probability <1 (augmentation-style) instead of always-on, to isolate whether this is indeed the cause of the observed in-domain cost.
- Combining aggressive augmentation and appearance adaptation, to check whether the effects add up.
- Self-training with pseudo-labels on the target domain (transductive setting, not yet attempted in this work).
- An ensemble extended to folds as well, not just architectures.
- Investigating the cause of the consistently observed higher difficulty of HemoSet fold 3.
- When available, repeating the most promising experiments (aggressive augmentation + ensemble first) on real TRAMIS data, keeping in mind that mixing external datasets of very different sizes into training is not automatically beneficial without explicit domain balancing.
- Evaluating extensions based on pretrained Transformer backbones (e.g., DINOv3, as in the BorDINO model proposed by Giusti & Marzo), outside the CNN-based scope of this work.

---

## 7. Conclusions

This work systematically quantified the asymmetric domain gap between HemoSet and Rabbani, and compared six families of strategies for increasing the cross-dataset robustness of surgical blood segmentation models. **Aggressive data augmentation** proved to be the most effective and lowest-cost strategy among those tested; combined with an **ensemble of three architectures** (UNet, UNet++, DeepLabV3+), it produces the best overall result, with no additional training beyond what augmentation already requires. Three further extensions did not improve on this result, but their analysis supports the hypothesis that the key factor is preserving the model's exposure to the "clean" distribution of its own domain during training, rather than imposing invariance through signals or normalizations computed directly on the data.

---

## 8. References

1. A. J. Miao, S. Lin, J. Lu, F. Richter, B. Ostrander, E. K. Funk, R. K. Orosco, M. C. Yip. "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management." *International Symposium on Medical Robotics (ISMR)*, 2024.
2. N. Rabbani, C. Seve, N. Bourdel, A. Bartoli. "Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation." *Medical Imaging with Deep Learning (MIDL)*, 2022.
3. Y. Yang, S. Soatto. "FDA: Fourier Domain Adaptation for Semantic Segmentation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
4. E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley. "Color Transfer between Images." *IEEE Computer Graphics and Applications*, 2001.
5. K. Zhou, Y. Yang, Y. Qiao, T. Xiang. "Domain Generalization with MixStyle." *International Conference on Learning Representations (ICLR)*, 2021.
6. Y. Li, N. Wang, J. Shi, J. Liu, X. Hou. "Revisiting Batch Normalization For Practical Domain Adaptation." *arXiv:1603.04779*, 2016.
7. Su et al. *AAAI Conference on Artificial Intelligence*, 2023. (single-source domain generalization via augmentation)
8. F. Giusti, F. Marzo. Project report, Computer Vision and Cognitive Systems, UNIMORE, A.Y. 2025/2026.
