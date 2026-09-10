# BleedSense: Cross-Dataset Robustness for Surgical Blood Segmentation

**Final report — Computer Vision and Cognitive Systems (CVCS) course, A.Y. 2025/2026, UNIMORE**

Riccardo Mazzi
Supervisor: Dr. Livia Del Gaudio (AImageLab, TRAMIS project)

---

## Abstract

Automatic hemostasis management in robotic surgery requires reliable segmentation of blood in the operative field. Models trained on a single surgical dataset, however, generalize poorly to other datasets with different visual characteristics. This work studies the problem on two public datasets, HemoSet (robotic surgery on a porcine model) and Rabbani et al. (gynecological laparoscopy on humans), confirming an asymmetric domain gap already observed in previous work from the research group, and systematically compares six families of strategies to reduce it: aggressive data augmentation, appearance/domain adaptation (Reinhard, FDA), multi-domain joint training (with and without balancing), ensembling of multiple architectures, an illumination-invariant input channel and feature-level style mixing (MixStyle), plus a test-time BatchNorm recalibration. Across three standard architectures (UNet, UNet++, DeepLabV3+, `resnet34` encoder), the most effective strategy is aggressive data augmentation, which improves cross-dataset robustness in both directions with no appreciable in-domain cost; combined with an ensemble of the three architectures, it yields the best overall result. Three further, more original extensions attempted afterwards did not improve on this result any further, but their analysis provides useful insight into why more targeted interventions fail in this setting.

---

## 1. Introduction and Motivation

Automating hemostasis management in robotic surgery — identifying and controlling bleeding during the procedure — requires, as a first step, reliable segmentation of blood in the endoscopic field of view. This project is part of the TRAMIS research initiative, focused on real-time bleeding detection in robotic surgery; since real, annotated TRAMIS surgical data were not yet available at the time of this work, the study was conducted on two public reference datasets in the domain, used as proxies:

- **HemoSet**: induced bleeding during teleoperated robotic surgery on a porcine model, 962 image-mask pairs, 10 subjects.
- **Rabbani et al.**: gynecological laparoscopy on human patients, 751 annotated images.

Previous work from the research group had observed an *asymmetric* performance drop when a model trained on one of the two datasets is evaluated on the other: HemoSet→Rabbani shows a strong general performance drop, while Rabbani→HemoSet shows a marked increase in false positives (non-hemorrhagic regions segmented as blood).

**Project goal**: to quantify this asymmetry rigorously and reproducibly, and to systematically study which strategies actually increase the cross-dataset robustness of segmentation models, comparing multiple experimental conditions without the constraint of converging on a single solution.

---

## 2. Related Work

Miao et al. introduced the HemoSet dataset and benchmarked standard segmentation models with IoU/F1/Hausdorff Distance metrics. Rabbani et al. proposed a dataset and segmentation system for laparoscopic bleeding with an adversarial domain adaptation component, evaluated with IoU and F-Score. Yang and Soatto introduced Fourier Domain Adaptation (FDA), swapping low-frequency amplitude components in the Fourier spectrum, used here as an appearance adaptation method (Section 4.2). Reinhard et al. proposed statistical color transfer in L\*a\*b\* space, the second appearance adaptation method used. Zhou et al. proposed MixStyle, feature-level style mixing for domain generalization, used here as an extension (Section 5.2). Li et al. proposed AdaBN, test-time BatchNorm recalibration for domain adaptation, also used as an extension. Su et al. studied single-source domain generalization via augmentation, informing the interpretation of our augmentation results. Teevno et al. proposed style-content disentanglement for domain generalization in endoscopic segmentation, a more architecture-heavy alternative to the strategies compared in this work. Finally, Giusti and Marzo, in a previous project from the same course, first quantified the asymmetric domain gap between HemoSet and Rabbani (zero-shot Dice: HemoSet→Rabbani 0.29, Rabbani→HemoSet 0.56), qualitatively consistent with our Section 4.1 despite different splits and hyperparameters.

Relative to Rabbani et al.'s own adversarial domain adaptation module and Teevno et al.'s style-content disentanglement, both of which require dedicated architectural components, this work deliberately favors simpler, more easily reproducible interventions — augmentation, color/frequency-based appearance transfer, and ensembling — to first isolate which basic mechanisms actually drive cross-dataset robustness, before layering on more complex architectural solutions.

---

## 3. Approach

### 3.1 Datasets and Splits

HemoSet consists of 962 image-mask pairs from 10 subjects (pigs); we use a Stratified Group K-Fold split into 5 folds, with the subject as the group, to avoid the same animal appearing in both training and validation within the same fold. Rabbani consists of 751 images/masks with a fixed 70/15/15 split (train/val/test, seed 42), reused identically across all phases of the project.

### 3.2 Architectures and Training Protocol

We use three standard architectures via `segmentation_models_pytorch`, all with the same `resnet34` encoder for a fair comparison: UNet, UNet++, and DeepLabV3+. Training uses a combined Dice+BCE loss, the AdamW optimizer (lr 1e-4), a `ReduceLROnPlateau` scheduler, batch size 8, 256×256 images, and the full dataset for each split. Early stopping on validation Dice is used (patience 8, cap of 40 epochs) — we empirically verified, across all 87 Phase 1–4 runs, that this budget was sufficient for convergence (only 2/87 runs reached the cap, both with a residual improvement below 0.01 Dice). The exact same configuration is used for every condition compared, to isolate the effect of the single variable under study.

### 3.3 Metrics and Evaluation Protocol

We report Dice, IoU, Precision, Recall, F1, and HD95 (95th-percentile Hausdorff Distance). This set fully covers the metrics used in the original papers of both datasets, with HD95 preferred over the maximum HD because it is less sensitive to single-pixel outliers. A model trained on HemoSet is evaluated in-domain on the validation fold and cross-dataset on the fixed Rabbani test set; a model trained on Rabbani is evaluated in-domain on its own fixed test set and cross-dataset on the whole of HemoSet (never seen during training). We note that this makes the two datasets' results statistically asymmetric: HemoSet figures are mean ± standard deviation over 5 folds, while Rabbani (which has no natural grouping variable to fold over) contributes a single run — Rabbani numbers should therefore be read with correspondingly more caution.

---

## 4. Results

### 4.1 Baseline and Domain Gap Asymmetry

The three architectures, trained separately on the two datasets with minimal augmentation, confirm and quantify the asymmetry: for HemoSet→Rabbani, mean Dice drops by about 70% relative (0.745 → 0.217), with HD95 rising from ~29px to ~130px, driven by a collapse in recall (0.76 → 0.19): the model becomes too conservative on the target domain, missing most of the actual blood. For Rabbani→HemoSet, the drop is more contained (0.684 → 0.585) but precision collapses (0.74 → 0.50) while recall stays high — consistent with the original observation about false positives. The three architectures behave very similarly under every condition: none resolves the domain gap on its own.

### 4.2 Data Augmentation, Appearance Adaptation, Joint Training

An "aggressive" augmentation (hue/saturation, gamma, blur, noise, JPEG compression, via Albumentations) improves cross-dataset robustness across all three architectures in both directions, at almost no in-domain cost (HemoSet→Rabbani: 0.217 → 0.321, +48% relative; Rabbani→HemoSet: 0.585 → 0.601). Reinhard color transfer and FDA, applied using unlabeled target-domain images, help the more critical direction (HemoSet→Rabbani: 0.217 → 0.263 and → 0.242 respectively) but at a non-negligible in-domain cost (Rabbani in-domain: 0.684 → 0.611 with Reinhard) and do not beat aggressive augmentation. We hypothesize that, unlike augmentation (probabilistic, applied with some probability per image), adaptation here is always active during training: the model never sees the "clean" appearance of its own source domain, and ends up split between two styles instead of properly consolidating one with variations.

Training on the union of HemoSet and Rabbani (joint training, with separate held-out test sets) was not a solution: no systematic benefit on HemoSet, and a systematic drop on Rabbani (0.684 → 0.636) caused by the size imbalance between the two datasets in the combined training set (~59% HemoSet vs ~41% Rabbani). A domain-balanced sampling scheme (`WeightedRandomSampler`, ~50/50 per batch) recovers only about half of the drop (0.636 → 0.656), without making joint training competitive with the other strategies.

### 4.3 Ensemble of the Three Architectures

As an original contribution beyond the assigned roadmap, we ensembled the three architectures' predictions at zero additional training cost, simply averaging the sigmoid probabilities of the already-trained aggressive-augmentation checkpoints. The result beats the best single model on all four tested conditions, with modest but systematic, exception-free gains (e.g., HemoSet→Rabbani: 0.337 best single model → 0.343 ensemble; Rabbani→HemoSet: 0.615 → 0.623). This is consistent with the standard ensembling rationale: errors made independently by architecturally different models tend to disagree and partially cancel out when their probabilities are averaged, while the shared, correct signal is reinforced.

A practical caveat is worth noting given this project's real-time motivation (Section 1): the ensemble triples inference cost (three forward passes instead of one) relative to a single model. This is a favorable trade-off for the offline evaluation performed here, but a deployed real-time bleeding-detection system would need to weigh the modest gains reported above against this added latency, or fall back to aggressive augmentation alone — which is already the single most effective and cheapest strategy tested — if strict real-time constraints apply.

### 4.4 Comparative Summary

Table 1 reports the mean over the three architectures for each strategy; full per-architecture results are in the supplementary materials.

**Table 1.** Mean Dice by strategy and evaluation condition.

| Strategy | HemoSet (in-domain) | Rabbani (in-domain) | HemoSet→Rabbani (cross) | Rabbani→HemoSet (cross) |
|---|---|---|---|---|
| Baseline (source-only) | 0.745 | 0.684 | 0.217 | 0.585 |
| Aggressive augmentation | 0.747 | 0.681 | 0.321 | 0.601 |
| Adaptation (Reinhard) | 0.719 | 0.611 | 0.263 | 0.577 |
| Adaptation (FDA) | 0.720 | 0.675 | 0.242 | 0.550 |
| Natural joint training | 0.742 | 0.636 | n/a* | n/a* |
| Balanced joint training | 0.739 | 0.656 | n/a* | n/a* |
| **Ensemble (3 arch.)** | **0.770** | **0.706** | **0.343** | **0.623** |

\* joint training is evaluated in-domain on both test sets, so there is no comparable cross-dataset condition for this row.

*(Figure 1, the same chart shown in the PDF version of this report, plots this table's mean Dice by strategy and evaluation condition.)*

**Recommended configuration**: aggressive augmentation in training + ensemble of the three architectures at inference — the only combination that improves cross-dataset robustness in both directions without penalizing in-domain performance.

### 4.5 Qualitative Results

Figures 2 and 3 (see PDF rendering of this report) show representative examples for each cross-dataset direction, comparing the ground-truth mask against the predictions of the source-only baseline, the aggressive-augmentation model, and the ensemble, overlaid on the original image. The baseline under-segmentation on HemoSet→Rabbani (missed blood regions) and the over-segmentation on Rabbani→HemoSet (false-positive regions) are visually apparent, and progressively mitigated by aggressive augmentation and further by the ensemble.

---

## 5. Discussion

### 5.1 Why Augmentation Beats Adaptation and Joint Training

The common factor behind the two less effective strategies — always-active appearance adaptation, imbalanced joint training — appears to be the lack of a mechanism that still preserves the model's exposure to the "clean" distribution of its own source domain during training. A light, probabilistic intervention on visual variety beats more targeted but always-active interventions on style, or simply mixing the data together. This is visually corroborated by the qualitative examples in Figures 2–3: aggressive augmentation and the ensemble progressively recover the recall lost by the source-only baseline on HemoSet→Rabbani, and reduce — without fully eliminating — the over-segmentation observed in the opposite direction.

### 5.2 Three Further Extensions (Negative but Informative Outcome)

After identifying the recommended configuration, we attempted three additional original extensions:

1. **"Blood-index" input channel** (ratio R/(R+G+B), in theory less sensitive to camera/illumination differences): slightly improves HemoSet→Rabbani (0.321 → 0.331) but markedly worsens Rabbani→HemoSet (0.601 → 0.526). The index, computed on raw pixels, carries over the white-balance differences between the datasets instead of ignoring them, becoming a shortcut feature specific to the source domain.
2. **MixStyle**: mixes the encoder's feature-map statistics between samples in the same batch. A diffuse but small degradation in both cross-dataset directions (0.217 → 0.203; 0.585 → 0.561), likely because it is applied too early in the encoder and because training batches remain single-domain in this setting, limiting the actual style diversity injected.
3. **Test-time BatchNorm recalibration** (AdaBN): recalibrates BatchNorm statistics on unlabeled target-domain images at inference time, with no additional training. Shifts the precision/recall trade-off in opposite directions in the two cross-dataset directions but net Dice does not improve in either direction (0.321 → 0.318; 0.601 → 0.570).

None of the three beats the recommended configuration. The pattern is nonetheless informative: three very different mechanisms fail in the same way — either introducing a shortcut specific to the source domain, or disturbing useful information without replacing it with something more transferable — reinforcing the conclusion that aggressive augmentation + ensemble is not a local optimum that can be easily improved with light interventions, at least for this pair of datasets.

---

## 6. Limitations and Future Work

- Repeating the Rabbani runs with multiple seeds (or an alternative resampling scheme, since Rabbani has no natural grouping variable to fold over as HemoSet does) to obtain a variance estimate comparable to HemoSet's 5-fold results.
- Applying appearance/domain adaptation with probability <1 instead of always-on, to isolate whether this is indeed the cause of the observed in-domain cost.
- Combining aggressive augmentation and appearance adaptation, to check whether the effects add up.
- Self-training with pseudo-labels on the target domain (transductive setting, not yet attempted).
- Style-content disentanglement (Teevno et al.), a more architecture-heavy alternative not explored here.
- An ensemble extended to folds as well, not just architectures.
- Investigating the cause of the consistently observed higher difficulty of HemoSet fold 3.
- Repeating the most promising experiments on real TRAMIS data when available, keeping in mind that mixing external datasets of very different sizes into training is not automatically beneficial without explicit domain balancing.
- Evaluating extensions based on pretrained Transformer backbones (e.g., DINOv3, as in the BorDINO model proposed by Giusti and Marzo), outside the CNN-based scope of this work.

---

## 7. Conclusions

This work systematically quantified the asymmetric domain gap between HemoSet and Rabbani, and compared six families of strategies for increasing the cross-dataset robustness of surgical blood segmentation models. Aggressive data augmentation proved to be the most effective and lowest-cost strategy among those tested; combined with an ensemble of three architectures, it produces the best overall result, with no additional training beyond what augmentation already requires. Three further extensions did not improve on this result, but their analysis supports the hypothesis that the key factor is preserving the model's exposure to the "clean" distribution of its own domain during training, rather than imposing invariance through signals or normalizations computed directly on the data.

---

## References

1. A. J. Miao, S. Lin, J. Lu, F. Richter, B. Ostrander, E. K. Funk, R. K. Orosco, M. C. Yip, "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management," *International Symposium on Medical Robotics (ISMR)*, 2024.
2. N. Rabbani, C. Seve, N. Bourdel, A. Bartoli, "Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation," *Medical Imaging with Deep Learning (MIDL)*, 2022.
3. Y. Yang, S. Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation," *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
4. E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley, "Color Transfer between Images," *IEEE Computer Graphics and Applications*, 2001.
5. K. Zhou, Y. Yang, Y. Qiao, T. Xiang, "Domain Generalization with MixStyle," *International Conference on Learning Representations (ICLR)*, 2021.
6. Y. Li, N. Wang, J. Shi, J. Liu, X. Hou, "Revisiting Batch Normalization For Practical Domain Adaptation," *arXiv:1603.04779*, 2016.
7. H. Su et al., "Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation," *AAAI Conference on Artificial Intelligence*, 2023.
8. M. A. Teevno et al., "Domain Generalization for Endoscopic Image Segmentation by Disentangling Style-Content Information and SuperPixel Consistency," *IEEE International Symposium on Computer-Based Medical Systems (CBMS)*, 2024.
9. F. Giusti, F. Marzo, Project report, Computer Vision and Cognitive Systems, UNIMORE, A.Y. 2025/2026.
