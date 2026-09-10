# BleedSense — Precision Hemostasis

Individual project for the Computer Vision and Cognitive Systems (CVCS) course, A.Y. 2025/2026, UNIMORE. Supervisor: Dr. Livia Del Gaudio (AImageLab, TRAMIS project).

## Goal

Study strategies to increase the **cross-dataset robustness** of surgical blood segmentation models, comparing two public datasets:

- **HemoSet** (Miao et al., ISMR 2024) — robotic surgery on pigs, 10 subjects, no official split.
- **Rabbani et al.** (MIDL 2022) — gynecological laparoscopy on humans.

Prior work from the research group observed an asymmetric performance drop when a model trained on one dataset is evaluated on the other: HemoSet→Rabbani shows a general performance drop, while Rabbani→HemoSet shows a strong increase in false positives. This project systematically compares augmentation, appearance/domain adaptation, joint training, ensembling, and a few additional extensions (input channel, feature-level style mixing, test-time normalization) to reduce this gap.

**Report**: [`docs/cvcs2026_bleedsense_final_report.md`](docs/cvcs2026_bleedsense_final_report.md) / [`docs/cvcs2026_bleedsense_report.html`](docs/cvcs2026_bleedsense_report.html) (main report, with a summary chart) and [`docs/cvcs2026_bleedsense_supplementary.html`](docs/cvcs2026_bleedsense_supplementary.html) (full per-architecture result tables).

## Repository structure

```
bleedsense/
├── docs/
│   ├── cvcs2026_bleedsense_final_report.md    # final report source (Markdown)
│   ├── cvcs2026_bleedsense_report.html        # printable version of the report, with summary chart + qualitative figures
│   ├── cvcs2026_bleedsense_supplementary.html # full per-architecture result tables
│   ├── bleedsense_report.tex                  # LaTeX (IEEEtran, two-column) version — build on Overleaf for the submitted PDF
│   └── reference_papers/                      # HemoSet, Rabbani, FDA, prior group report (PDF)
├── results/                 # raw per-run JSON results + aggregated summary tables (.md)
│   └── qualitative/          # example segmentation overlays (ground truth vs. predictions), used in the report figures
├── dataset.py               # HemoSet + Rabbani loaders, Stratified Group K-Fold, transforms
├── models.py                 # architecture factory (segmentation_models_pytorch) + MixStyle
├── domain_adaptation.py       # Reinhard color transfer, Fourier Domain Adaptation (FDA)
├── metrics.py                 # Dice, IoU, Precision/Recall/F1, HD95, MetricTracker
├── train.py                    # training/evaluation entry point (all phases + extensions) — the actual reproducible unit
├── train.sh                     # SLURM wrapper around train.py (AImageLab-HPC-specific, see note below)
├── submit_phase1_baseline.sh     # baseline: 3 architectures x (5 HemoSet folds + Rabbani)
├── submit_phase2_augmentation.sh  # aggressive augmentation condition
├── submit_phase3_adaptation.sh     # Reinhard / FDA appearance adaptation
├── submit_phase4_joint.sh           # joint training (natural domain sampling)
├── submit_phase4b_joint_balanced.sh  # joint training (balanced domain sampling)
├── submit_extra_bloodindex.sh         # extension: blood-index input channel
├── submit_extra_mixstyle.sh            # extension: MixStyle
├── ensemble_eval.py                     # extension: ensemble of the 3 architectures (inference only)
├── test_time_bn.py                       # extension: test-time BatchNorm recalibration (inference only)
├── visualize_predictions.py               # generates the qualitative report figures (inference only)
├── aggregate_results.py                   # builds results/results_summary.md from the JSON files
├── aggregate_joint_results.py              # same, for the joint-training runs (Phase 4/4b)
├── check_convergence.py                     # verifies the training budget (epochs/patience) was adequate
├── test_dataset.py                            # sanity check for the data loaders
└── requirements.txt
```

`datasets/`, `results/checkpoints/`, and `logs/` are git-ignored (raw data, model weights, and SLURM logs — too large / not ours to redistribute). The `results/*.json`, `results/*.md`, and `results/qualitative/*.png` files themselves **are** committed: they are the raw/aggregated outputs of every run and are the evidence backing every table and figure in the report.

**Important**: `train.sh` and the `submit_*.sh` scripts are specific to the AImageLab-HPC SLURM cluster (they `sbatch` jobs, `module load` cluster modules, and `cd` into a fixed `/work/cvcs2026/bleedsense` path) — they will **not** run as-is outside that account. `train.py` itself has no such dependency: on any machine with a GPU (or CPU) and the datasets downloaded locally, call it directly with `--data-dir`/`--output-dir` pointing at local paths — see "Reproducing a single run without SLURM" below. The `submit_*.sh` files are simple loops calling `train.py` with different flag combinations; open one to see the exact flags used for that phase.

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

All runs share the same training configuration (Dice+BCE loss, AdamW lr 1e-4, `ReduceLROnPlateau`, batch size 8, 256×256 images, early stopping patience 8, cap 40 epochs — see `train.py` argparse defaults) unless a flag says otherwise. The table below shows how each phase was launched on the AImageLab-HPC cluster (SLURM); if you don't have access to that cluster, skip straight to "Reproducing a single run without SLURM" below instead.

| Phase | What it tests | Launch (on the AImageLab-HPC cluster) | Output files |
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
| Extra — Qualitative figures | inference only, needs the Phase 1/2 checkpoints above | `python visualize_predictions.py` | `results/qualitative/*.png` |

After the JSON files for a phase exist under `results/`, rebuild the aggregated Markdown tables with:

```bash
python aggregate_results.py          # Phases 1-3 + blood-index/MixStyle extensions -> results/results_summary.md
python aggregate_joint_results.py    # Phase 4/4b -> results/phase4_joint_summary.md
```

`check_convergence.py` re-reads the per-epoch history stored in each result JSON and flags any run that hit the 40-epoch cap while still improving by more than 0.01 Dice (used to validate that the training budget was adequate — see the report, Section 3.2).

## Reproducing a single run without SLURM

Every `submit_*.sh` script is only a loop of `sbatch ... train.sh <args>` calls for the AImageLab-HPC cluster. To reproduce any individual run on a plain machine with a GPU (no SLURM needed), call `train.py` directly with local paths, e.g. the very first Phase 1 baseline run:

```bash
python train.py \
    --dataset hemoset --architecture unet --fold 0 \
    --data-dir ./datasets --output-dir ./results
```

or the Phase 2 aggressive-augmentation condition on Rabbani:

```bash
python train.py \
    --dataset rabbani --architecture unet --augmentation aggressive \
    --data-dir ./datasets --output-dir ./results
```

The full set of flags for any phase/extension is visible by opening the corresponding `submit_*.sh` file — each is a short, readable loop, e.g. `submit_phase2_augmentation.sh` just calls the command above for the 3 architectures and 5 HemoSet folds plus Rabbani. A quick end-to-end sanity check with a tiny epoch budget (useful before committing to a full 40-epoch run):

```bash
python train.py --dataset hemoset --architecture unet --fold 0 --epochs 2 \
    --data-dir ./datasets --output-dir ./results
```

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
