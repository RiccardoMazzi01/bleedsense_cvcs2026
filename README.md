# BleedSense — Precision Hemostasis

Individual project for the Computer Vision and Cognitive Systems (CVCS) course, A.Y. 2025/2026, UNIMORE. Supervisor: Dr. Livia Del Gaudio (AImageLab, TRAMIS project).

## Goal

Study strategies to increase the **cross-dataset robustness** of surgical blood segmentation models, comparing two public datasets:

- **HemoSet** (Miao et al., ISMR 2024) — robotic surgery on pigs, 10 subjects, no official split.
- **Rabbani et al.** (MIDL 2022) — gynecological laparoscopy on humans.

Prior work from the research group observed an asymmetric performance drop when a model trained on one dataset is evaluated on the other: HemoSet→Rabbani shows a general performance drop, while Rabbani→HemoSet shows a strong increase in false positives. This project systematically compares augmentation, appearance/domain adaptation, joint training, ensembling, and a few additional extensions (input channel, feature-level style mixing, test-time normalization) to reduce this gap.

**Full narrative report** (all decisions, experiments, and results, in order): [`docs/cvcs2026_bleedsense_report.md`](docs/cvcs2026_bleedsense_report.md) (Italian, working log). Course deliverables (English): [`docs/cvcs2026_bleedsense_final_report.md`](docs/cvcs2026_bleedsense_final_report.md) / [`docs/cvcs2026_bleedsense_report.html`](docs/cvcs2026_bleedsense_report.html) (main report) and [`docs/cvcs2026_bleedsense_supplementary.html`](docs/cvcs2026_bleedsense_supplementary.html) (full per-architecture tables).

## Repository structure

```
bleedsense/
├── docs/
│   ├── cvcs2026_bleedsense_report.md          # full chronological log (source of truth, Italian)
│   ├── cvcs2026_bleedsense_final_report.md    # concise final report (English, course deliverable)
│   ├── cvcs2026_bleedsense_report.html        # printable version of the final report, with summary chart
│   ├── cvcs2026_bleedsense_supplementary.html # full per-architecture result tables (English)
│   ├── reference_papers/                      # HemoSet, Rabbani, FDA, prior group report (PDF)
│   └── correspondence/                        # drafts/updates sent to the supervisor (Italian)
├── results/                # raw per-run JSON results + aggregated summary tables (.md)
├── dataset.py               # HemoSet + Rabbani loaders, Stratified Group K-Fold, transforms
├── models.py                 # architecture factory (segmentation_models_pytorch) + MixStyle
├── domain_adaptation.py       # Reinhard color transfer, Fourier Domain Adaptation (FDA)
├── metrics.py                 # Dice, IoU, Precision/Recall/F1, HD95, MetricTracker
├── train.py                    # training/evaluation entry point (all phases + extensions)
├── train.sh                     # SLURM submission script wrapping train.py
├── submit_phase1_baseline.sh     # baseline: 3 architectures x (5 HemoSet folds + Rabbani)
├── submit_phase2_augmentation.sh  # aggressive augmentation condition
├── submit_phase3_adaptation.sh     # Reinhard / FDA appearance adaptation
├── submit_phase4_joint.sh           # joint training (natural domain sampling)
├── submit_phase4b_joint_balanced.sh  # joint training (balanced domain sampling)
├── submit_extra_bloodindex.sh         # extension: blood-index input channel
├── submit_extra_mixstyle.sh            # extension: MixStyle
├── ensemble_eval.py                     # extension: ensemble of the 3 architectures (inference only)
├── test_time_bn.py                       # extension: test-time BatchNorm recalibration (inference only)
├── aggregate_results.py                   # builds results/results_summary.md from the JSON files
├── aggregate_joint_results.py              # same, for the joint-training runs (Phase 4/4b)
├── check_convergence.py                     # verifies the training budget (epochs/patience) was adequate
├── test_dataset.py                            # sanity check for the data loaders
└── requirements.txt
```

`datasets/`, `results/checkpoints/`, and `logs/` are git-ignored (raw data, model weights, and SLURM logs — too large / not ours to redistribute). The `results/*.json` and `results/*.md` files themselves **are** committed: they are the raw/aggregated outputs of every run and are the evidence backing every table in the report.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install torch/torchvision first, matching your CUDA driver, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# (or a plain `pip install torch torchvision` / CPU build on a machine without that constraint)

pip install -r requirements.txt
```

## Dataset layout

Datasets are **not included** in the repository (see `.gitignore`) and must be downloaded separately:

- HemoSet — Miao et al., ISMR 2024: [Google Drive folder](https://drive.google.com/drive/folders/19-2cHwDslxjDelMioHvTV1ToFKzOAqbd).
- Rabbani et al. — MIDL 2022: [dataset archive](https://web.archive.org/web/20220705182845if_/http://igt.ip.uca.fr/~ab/code_and_datasets/datasets/bleeding_segmentation_v1p0.zip) (mirrored via the Wayback Machine; the original host is no longer reliably reachable).

Place them under `datasets/` (path configurable via `--data-dir`, default `/work/cvcs2026/bleedsense/datasets` for cluster runs) with this layout, expected by `dataset.py`:

```
datasets/
├── hemoset/
│   └── pig<N>/
│       ├── images/*.png|*.jpg
│       └── masks/*.png|*.jpg      # same filename as the image, or with a _mask suffix
└── rabbani/
    ├── images/*
    └── masks/*
```

The HemoSet Stratified Group K-Fold split (group = pig, seed 42) and the Rabbani fixed 70/15/15 split (seed 42) are computed on the fly from this layout — no split files to download, they are reproduced identically on any machine from the same seed.

## Reproducing the experiments

All runs share the same training configuration (Dice+BCE loss, AdamW lr 1e-4, `ReduceLROnPlateau`, batch size 8, 256×256 images, early stopping patience 8, cap 40 epochs — see `train.py` argparse defaults) unless a flag says otherwise. Every submit script below is a thin loop over `sbatch ... train.sh <args>` (SLURM); the same runs can be reproduced without SLURM by calling `python train.py <args>` directly (see `train.sh` for the exact argument mapping) or `bash train.sh <args>` on a machine with a GPU.

| Phase | What it tests | Launch | Output files |
|---|---|---|---|
| 1 — Baseline | 3 architectures × (5 HemoSet folds + Rabbani), light augmentation | `bash submit_phase1_baseline.sh` | `results/hemoset_<arch>_fold<N>.json`, `results/rabbani_<arch>.json` |
| 2 — Augmentation | same, aggressive augmentation | `bash submit_phase2_augmentation.sh` | `..._augaggressive.json` |
| 3 — Appearance adaptation | same, light augmentation + Reinhard/FDA | `bash submit_phase3_adaptation.sh` | `..._adaptreinhard.json` / `..._adaptfda.json` |
| 4 — Joint training | 3 architectures × 5 folds, HemoSet+Rabbani combined training set | `bash submit_phase4_joint.sh` | `results/joint_<arch>_fold<N>.json` |
| 4b — Balanced joint training | same, `WeightedRandomSampler` | `bash submit_phase4b_joint_balanced.sh` | `..._balanced.json` |
| Extra — Blood-index channel | on top of the aggressive condition, 4th input channel | `bash submit_extra_bloodindex.sh` | `..._augaggressive_bloodindex.json` |
| Extra — MixStyle | on top of the light/none condition | `bash submit_extra_mixstyle.sh` | `..._mixstyle.json` |
| Extra — Ensemble | inference only, no new training | `python ensemble_eval.py` | `results/ensemble_aggressive.json` + `.md` |
| Extra — Test-time BN recalibration | inference only, no new training | `python test_time_bn.py` | `results/test_time_bn_aggressive.json` + `.md` |

After the JSON files for a phase exist under `results/`, rebuild the aggregated Markdown tables with:

```bash
python aggregate_results.py          # Phases 1-3 + blood-index/MixStyle extensions -> results/results_summary.md
python aggregate_joint_results.py    # Phase 4/4b -> results/phase4_joint_summary.md
```

`check_convergence.py` re-reads the per-epoch history stored in each result JSON and flags any run that hit the 40-epoch cap while still improving by more than 0.01 Dice (used to validate that the training budget was adequate — see the report, Section 3.2).

## Key CLI flags of `train.py`

```
--dataset {hemoset,rabbani,joint}
--architecture {unet,unetplusplus,deeplabv3plus}
--fold N                        # 0-4, for hemoset/joint
--augmentation {light,aggressive}
--adaptation {none,reinhard,fda}
--joint-sampling {natural,balanced}   # only with --dataset joint
--use-blood-index               # 4-channel input (R/(R+G+B) extra channel)
--mixstyle                      # feature-level style mixing on the encoder
```

All flags default to the Phase 1 baseline behavior (backward compatible): each extension is opt-in via an explicit flag, never changes the behavior of earlier phases.
