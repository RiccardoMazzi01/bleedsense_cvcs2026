# BleedSense — Precision Hemostasis

Progetto individuale del corso CVCS 2025/2026 (Computer Vision and Cognitive Systems, UNIMORE), tutor: Dott.ssa Livia Del Gaudio (AImageLab, progetto TRAMIS).

## Obiettivo

Studiare strategie per aumentare la **robustezza cross-dataset** di modelli di segmentazione del sangue in ambito chirurgico, confrontando:

- **HemoSet** (Miao et al., ISMR 2024) — chirurgia robotica su maiali, split predefiniti assenti, 10 soggetti (pig).
- **Rabbani et al.** (MIDL 2022) — laparoscopia ginecologica su umani.

È stato osservato (in un lavoro precedente del gruppo di ricerca) un calo di performance asimmetrico quando un modello allenato su un dataset viene testato sull'altro:
- HemoSet → Rabbani: calo generale delle performance.
- Rabbani → HemoSet: forte aumento dei falsi positivi.

Il progetto esplora, in fasi successive, quali strategie (augmentation, appearance/domain adaptation, joint training) riducono effettivamente questo domain shift.

## Struttura della repo

```
bleedsense/
├── docs/
│   ├── diario_di_bordo.md        # log dettagliato di tutte le attività svolte (report principale)
│   └── reference_papers/         # paper di riferimento (HemoSet, Rabbani, FDA, ecc.)
├── configs/                      # file di configurazione per i vari esperimenti (architettura, iperparametri, seed)
├── results/                      # tabelle/metriche/grafici prodotti dagli esperimenti
├── dataset.py                    # loader HemoSet + Rabbani, Stratified Group K-Fold
├── metrics.py                    # Dice, IoU, F1, HD95
├── models.py                     # factory per le architetture (segmentation_models_pytorch)
├── train.py                      # script di training/valutazione, lanciato via SLURM
└── train.sh                      # script di sottomissione SLURM
```

## Ambiente

- Cluster: AImageLab-HPC (`ailb-login-02/03.ing.unimore.it`)
- Codice ed esperimenti eseguiti su cluster via job SLURM; questa repo è il punto di sincronizzazione tra sviluppo locale e cluster (`git pull` sul cluster dopo ogni push).
- Ambiente Python: venv in `/homes/rmazzi/cvcs2026/venv`

Per il resoconto completo di ogni scelta, esperimento e risultato, vedi [docs/diario_di_bordo.md](docs/diario_di_bordo.md).
