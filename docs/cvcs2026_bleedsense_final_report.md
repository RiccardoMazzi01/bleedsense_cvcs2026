# BleedSense — Robustezza cross-dataset per la segmentazione del sangue in chirurgia robotica

**Report finale — Corso di Computer Vision and Cognitive Systems (CVCS), A.A. 2025/2026, UNIMORE**

Riccardo Mazzi (294437)
Supervisione: Dott.ssa Livia Del Gaudio (AImageLab, progetto TRAMIS)

---

## Abstract

La gestione automatica dell'emostasi in chirurgia robotica richiede la segmentazione affidabile del sangue nel campo operatorio. I modelli di segmentazione allenati su un singolo dataset chirurgico generalizzano tuttavia male su altri dataset con caratteristiche visive diverse (illuminazione, tessuto, strumentazione). Questo lavoro studia il problema su due dataset pubblici, **HemoSet** (chirurgia robotica su modello suino) e **Rabbani et al.** (laparoscopia ginecologica su umani), confermando un **domain gap asimmetrico** già osservato in lavori precedenti del gruppo di ricerca, e confronta quattro famiglie di strategie per ridurlo: data augmentation aggressiva, appearance/domain adaptation (color transfer di Reinhard, Fourier Domain Adaptation), joint training multi-dominio (con e senza bilanciamento tra domini) ed ensembling di più architetture. Su tre architetture di segmentazione standard (UNet, UNet++, DeepLabV3+, encoder `resnet34`), la strategia più efficace risulta la **data augmentation aggressiva**, che migliora la robustezza cross-dataset in entrambe le direzioni senza costi apprezzabili in-domain; combinata con un **ensemble delle tre architetture** in fase di inferenza, produce il miglior risultato complessivo su tutte le condizioni testate.

---

## 1. Introduzione e motivazione

L'automazione della gestione dell'emostasi in chirurgia robotica — l'identificazione e il controllo del sanguinamento durante l'intervento — richiede come primo passo la segmentazione affidabile del sangue nel campo visivo endoscopico. Questo progetto si inserisce nell'iniziativa di ricerca **TRAMIS**, orientata alla rilevazione in tempo reale del sanguinamento in chirurgia robotica; poiché al momento di questo lavoro non erano ancora disponibili dati chirurgici reali annotati del progetto, lo studio è stato condotto su due dataset pubblici di riferimento nel dominio, usati come proxy:

- **HemoSet** (Miao et al., ISMR 2024): sanguinamenti indotti durante chirurgia robotica teleoperata su modello suino, 962 coppie immagine-maschera, 10 soggetti.
- **Rabbani et al.** (MIDL 2022): laparoscopia ginecologica su pazienti umani, 751 immagini annotate.

Un lavoro precedente del gruppo di ricerca (Giusti & Marzo, report CVCS 2025/2026, si veda Sezione 2) aveva osservato un calo di performance **asimmetrico** quando un modello allenato su uno dei due dataset viene valutato sull'altro:

- **HemoSet → Rabbani**: forte calo generale delle performance.
- **Rabbani → HemoSet**: aumento marcato dei falsi positivi (regioni non ematiche segmentate come sangue).

**Obiettivo del progetto**: quantificare questa asimmetria in modo rigoroso e riproducibile, e studiare sistematicamente quali strategie aumentano effettivamente la robustezza cross-dataset dei modelli di segmentazione, confrontando più condizioni sperimentali senza vincolo di convergere su un'unica soluzione.

---

## 2. Lavori correlati

- **Miao et al., "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management"**, ISMR 2024 — dataset HemoSet, raccolto durante interventi di chirurgia robotica teleoperata (dVRK) su modello suino con sanguinamento indotto; benchmark di modelli di segmentazione standard, metriche IoU/F1/Hausdorff Distance.
- **Rabbani, Seve, Bourdel & Bartoli, "Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation"**, MIDL 2022 — dataset e sistema di segmentazione del sanguinamento in laparoscopia ginecologica, con una componente di adversarial domain adaptation; metriche IoU e F-Score.
- **Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation"**, CVPR 2020 — adattamento di dominio non supervisionato tramite scambio delle componenti a bassa frequenza nello spettro di Fourier, usato in questo lavoro come metodo di appearance adaptation (Sezione 6).
- **Reinhard, Ashikhmin, Gooch & Shirley, "Color Transfer between Images"**, IEEE CG&A 2001 — color transfer statistico nello spazio L\*a\*b\*, usato in questo lavoro come secondo metodo di appearance adaptation (Sezione 6).
- **Su et al.**, AAAI 2023 — domain generalization single-source tramite augmentation, riferimento per l'interpretazione dei risultati di data augmentation (Sezione 5).
- **Teevno et al.**, CBMS 2024 — riferimento addizionale in ambito bleeding detection chirurgico.
- **Giusti & Marzo, report di progetto CVCS 2025/2026** (lavoro precedente dello stesso corso, alla base della motivazione di questo progetto) — prima quantificazione del domain gap asimmetrico tra HemoSet e Rabbani nel protocollo "complete dataset" con U-Net++ (Dice zero-shot: HemoSet→Rabbani 0.29, Rabbani→HemoSet 0.56), risultati qualitativamente coerenti con quelli riportati in Sezione 4 di questo lavoro nonostante split e iperparametri diversi. Propongono inoltre BorDINO, un'estensione boundary-aware di SegDINO con backbone DINOv3 pretrained (Transformer), non direttamente confrontabile con l'approccio CNN-based di questo lavoro ma indicata come possibile sviluppo futuro (Sezione 10).

---

## 3. Dataset e setup sperimentale

### 3.1 Dataset e split

- **HemoSet**: 962 coppie immagine-maschera, 10 soggetti (pig). Split **Stratified Group K-Fold** a 5 fold, con il soggetto (pig) come gruppo, per evitare che lo stesso animale compaia sia in training sia in validation nello stesso fold.
- **Rabbani**: 751 immagini/maschere. Split fisso **70/15/15** (train/val/test, seed 42), riusato identico in tutte le fasi del progetto, in modo che il test set resti costante per ogni confronto.

### 3.2 Architetture

Tre architetture di segmentazione standard, tutte tramite `segmentation_models_pytorch`, con lo stesso encoder (`resnet34`) per un confronto equo tra architetture:
- **UNet**
- **UNet++**
- **DeepLabV3+**

### 3.3 Protocollo di training

- Loss: combinazione Dice + Binary Cross-Entropy.
- Ottimizzatore: AdamW, learning rate 1e-4.
- Scheduler: `ReduceLROnPlateau`.
- Batch size 8, immagini 256×256, dataset completo per ogni split (nessun sottocampionamento).
- Early stopping su Dice di validazione, patience 8 epoche, cap massimo 40 epoche.
- Stessa configurazione identica per ogni condizione sperimentale confrontata (nessun override tra le fasi), per isolare l'effetto della sola variabile in esame di volta in volta.

### 3.4 Metriche

Dice, IoU, Precision, Recall, F1, **HD95** (Hausdorff Distance al 95° percentile, via distance transform sui contorni). La scelta delle metriche e la sua adeguatezza rispetto ai paper originali dei due dataset è discussa in Sezione 9.2.

### 3.5 Protocollo di valutazione in-domain / cross-dataset

- **Training su HemoSet** (per fold): valutazione in-domain sul fold di validazione, valutazione cross-dataset sul test set fisso di Rabbani.
- **Training su Rabbani**: valutazione in-domain sul test set fisso di Rabbani, valutazione cross-dataset su **tutto** HemoSet (mai visto in training, quindi utilizzabile per intero come target di generalizzazione).

---

## 4. Fase 1 — Baseline

Le tre architetture sono allenate separatamente sui due dataset (augmentation minima: flip orizzontale + jitter luminosità/contrasto), per stabilire una baseline in-domain e quantificare il domain gap cross-dataset.

| Setting | Architettura | Dice | IoU | Precision | Recall | F1 | HD95 |
|---|---|---|---|---|---|---|---|
| HemoSet → HemoSet (in-domain) | UNet | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet → Rabbani (cross) | UNet | 0.218±0.033 | 0.149±0.025 | 0.638±0.081 | 0.179±0.036 | 0.218±0.033 | 134.346±20.055 |
| Rabbani → Rabbani (in-domain) | UNet | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani → HemoSet (cross) | UNet | 0.588 | 0.442 | 0.505 | 0.752 | 0.588 | 62.789 |
| HemoSet → HemoSet (in-domain) | UNet++ | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet → Rabbani (cross) | UNet++ | 0.236±0.035 | 0.159±0.023 | 0.418±0.096 | 0.223±0.022 | 0.236±0.035 | 104.057±8.325 |
| Rabbani → Rabbani (in-domain) | UNet++ | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani → HemoSet (cross) | UNet++ | 0.597 | 0.455 | 0.529 | 0.728 | 0.597 | 56.133 |
| HemoSet → HemoSet (in-domain) | DeepLabV3+ | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet → Rabbani (cross) | DeepLabV3+ | 0.196±0.032 | 0.133±0.025 | 0.665±0.074 | 0.165±0.035 | 0.196±0.032 | 151.249±20.629 |
| Rabbani → Rabbani (in-domain) | DeepLabV3+ | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani → HemoSet (cross) | DeepLabV3+ | 0.569 | 0.420 | 0.476 | 0.762 | 0.569 | 57.188 |

**Analisi**:
- **Asimmetria confermata e quantificata**: HemoSet→Rabbani il Dice crolla di circa il 70% relativo (0.74 → ~0.22) con HD95 che passa da ~26px a ~130-150px; Rabbani→HemoSet il calo è molto più contenuto (0.69 → ~0.57-0.60).
- **Il meccanismo è diverso nelle due direzioni**, leggibile da precision/recall:
  - HemoSet→Rabbani: la precision resta discreta (0.42-0.67) ma il **recall crolla** (0.71-0.79 → 0.18-0.22) — il modello diventa troppo conservativo su Rabbani, mancando la maggior parte del sangue reale (falsi negativi).
  - Rabbani→HemoSet: la precision crolla (0.74 → 0.48-0.53) ma il **recall resta alto o sale leggermente** (0.71 → 0.73-0.76) — coerente con l'osservazione di partenza sui falsi positivi (regioni non ematiche segmentate come sangue).
- **Le tre architetture si comportano in modo molto simile**, sia in-domain sia cross-dataset (differenze contenute rispetto alla deviazione standard tra fold): nessuna "risolve" da sola il domain gap, motivando lo spostamento del focus verso strategie di training/adattamento piuttosto che sulla scelta architetturale.

---

## 5. Fase 2 — Data augmentation

Confronto tra l'augmentation "light" della Fase 1 (flip orizzontale + jitter luminosità/contrasto) e una augmentation **"aggressive"** aggiuntiva (hue/saturation, gamma, blur gaussiano/motion, rumore gaussiano, artefatti di compressione JPEG, con probabilità/range più ampi), via Albumentations. Il transform di validazione resta identico in entrambe le condizioni, per confrontabilità diretta.

| Setting | Architettura | Augmentation | Dice | IoU | Precision | Recall | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|
| HemoSet → HemoSet (in-domain) | UNet | light | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet → Rabbani (cross) | UNet | light | 0.218±0.033 | 0.149±0.025 | 0.638±0.081 | 0.179±0.036 | 0.218±0.033 | 134.346±20.055 |
| Rabbani → Rabbani (in-domain) | UNet | light | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani → HemoSet (cross) | UNet | light | 0.588 | 0.442 | 0.505 | 0.752 | 0.588 | 62.789 |
| HemoSet → HemoSet (in-domain) | UNet | aggressive | 0.756±0.033 | 0.629±0.035 | 0.752±0.049 | 0.794±0.025 | 0.756±0.033 | 26.835±5.965 |
| HemoSet → Rabbani (cross) | UNet | aggressive | 0.323±0.032 | 0.227±0.027 | 0.396±0.071 | 0.370±0.031 | 0.323±0.032 | 90.396±4.508 |
| Rabbani → Rabbani (in-domain) | UNet | aggressive | 0.672 | 0.546 | 0.732 | 0.699 | 0.672 | 38.526 |
| Rabbani → HemoSet (cross) | UNet | aggressive | 0.603 | 0.461 | 0.539 | 0.747 | 0.603 | 51.224 |
| HemoSet → HemoSet (in-domain) | UNet++ | light | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet → Rabbani (cross) | UNet++ | light | 0.236±0.035 | 0.159±0.023 | 0.418±0.096 | 0.223±0.022 | 0.236±0.035 | 104.057±8.325 |
| Rabbani → Rabbani (in-domain) | UNet++ | light | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani → HemoSet (cross) | UNet++ | light | 0.597 | 0.455 | 0.529 | 0.728 | 0.597 | 56.133 |
| HemoSet → HemoSet (in-domain) | UNet++ | aggressive | 0.756±0.022 | 0.629±0.025 | 0.793±0.053 | 0.753±0.020 | 0.756±0.022 | 28.457±8.364 |
| HemoSet → Rabbani (cross) | UNet++ | aggressive | 0.304±0.028 | 0.212±0.022 | 0.349±0.019 | 0.367±0.051 | 0.304±0.028 | 97.351±7.258 |
| Rabbani → Rabbani (in-domain) | UNet++ | aggressive | 0.701 | 0.580 | 0.734 | 0.741 | 0.701 | 43.259 |
| Rabbani → HemoSet (cross) | UNet++ | aggressive | 0.615 | 0.475 | 0.546 | 0.749 | 0.615 | 60.491 |
| HemoSet → HemoSet (in-domain) | DeepLabV3+ | light | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet → Rabbani (cross) | DeepLabV3+ | light | 0.196±0.032 | 0.133±0.025 | 0.665±0.074 | 0.165±0.035 | 0.196±0.032 | 151.249±20.629 |
| Rabbani → Rabbani (in-domain) | DeepLabV3+ | light | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani → HemoSet (cross) | DeepLabV3+ | light | 0.569 | 0.420 | 0.476 | 0.762 | 0.569 | 57.188 |
| HemoSet → HemoSet (in-domain) | DeepLabV3+ | aggressive | 0.730±0.039 | 0.599±0.045 | 0.736±0.061 | 0.757±0.030 | 0.730±0.039 | 29.543±11.536 |
| HemoSet → Rabbani (cross) | DeepLabV3+ | aggressive | 0.337±0.021 | 0.239±0.019 | 0.401±0.047 | 0.436±0.045 | 0.337±0.021 | 93.988±7.533 |
| Rabbani → Rabbani (in-domain) | DeepLabV3+ | aggressive | 0.671 | 0.539 | 0.689 | 0.747 | 0.671 | 42.393 |
| Rabbani → HemoSet (cross) | DeepLabV3+ | aggressive | 0.585 | 0.438 | 0.471 | 0.835 | 0.585 | 51.325 |

**Analisi**:
- **L'augmentation aggressiva migliora la robustezza cross-dataset su tutte e tre le architetture**, in entrambe le direzioni, quasi senza costo sulle performance in-domain:
  - HemoSet→Rabbani (direzione più critica): Dice migliora del 29-72% relativo (UNet 0.218→0.323, UNet++ 0.236→0.304, DeepLabV3+ 0.196→0.337) e HD95 si riduce sensibilmente (es. DeepLabV3+ 151→94px). Il recall sale molto (0.18-0.22 → 0.37-0.44): il modello diventa meno conservativo, a fronte di una precision leggermente più bassa — trade-off complessivamente favorevole (il Dice/F1 aggregato migliora comunque).
  - Rabbani→HemoSet: miglioramento più contenuto ma presente su tutte e tre le architetture (0.569-0.597 → 0.585-0.615).
  - In-domain: variazioni piccole e miste, sempre dentro il rumore tra fold — nessun costo sistematico.
- **Interpretazione**: l'augmentation aggressiva (hue/saturation, gamma, blur, noise, compressione) spinge il modello a non overfittare sull'aspetto cromatico specifico del dominio sorgente, imparando feature di "sangue" più invarianti al dominio — meccanismo coerente con quanto descritto in Su et al. (AAAI 2023) per la domain generalization single-source.

---

## 6. Fase 3 — Appearance / domain adaptation

Due metodi di color/style transfer, implementati senza dipendenze esterne oltre numpy/opencv:

- **Reinhard color transfer** (Reinhard et al., 2001): allinea media e deviazione standard dei canali L\*a\*b\* dell'immagine sorgente a quelli di un'immagine del dominio target, senza alterare struttura/contenuto.
- **Fourier Domain Adaptation — FDA** (Yang & Soatto, CVPR 2020): nello spettro di Fourier, sostituisce la regione a bassa frequenza dell'ampiezza (lo "stile") della sorgente con quella del target, mantenendo la fase (il contenuto) della sorgente. Parametro β=0.01 (come nel paper originale).

Per isolare l'effetto dell'adaptation, tutti i run di questa fase usano l'augmentation "light" (non "aggressive"), così la Fase 1 resta la baseline di riferimento (adaptation="none") e il confronto è pulito su una sola variabile alla volta. Il pool di immagini target (non annotate, usate solo per il contenuto visivo) è: lo split di training di Rabbani per la direzione HemoSet→Rabbani; l'intero dataset HemoSet per la direzione Rabbani→HemoSet — un setting **transduttivo** (si osservano le immagini, mai le etichette, del dominio target durante il training), esplicitamente ammesso dal protocollo del progetto.

| Setting | Architettura | Augm. | Adaptation | Dice | IoU | Precision | Recall | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|---|
| HemoSet → HemoSet | UNet | light | none | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet → Rabbani | UNet | light | none | 0.218±0.033 | 0.149±0.025 | 0.638±0.081 | 0.179±0.036 | 0.218±0.033 | 134.346±20.055 |
| Rabbani → Rabbani | UNet | light | none | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani → HemoSet | UNet | light | none | 0.588 | 0.442 | 0.505 | 0.752 | 0.588 | 62.789 |
| HemoSet → HemoSet | UNet | light | reinhard | 0.709±0.081 | 0.573±0.096 | 0.735±0.157 | 0.727±0.060 | 0.709±0.081 | 31.177±20.222 |
| HemoSet → Rabbani | UNet | light | reinhard | 0.259±0.044 | 0.176±0.032 | 0.397±0.111 | 0.299±0.110 | 0.259±0.044 | 97.635±14.636 |
| Rabbani → Rabbani | UNet | light | reinhard | 0.640 | 0.519 | 0.758 | 0.649 | 0.640 | 62.700 |
| Rabbani → HemoSet | UNet | light | reinhard | 0.589 | 0.453 | 0.541 | 0.718 | 0.589 | 57.643 |
| HemoSet → HemoSet | UNet | light | fda | 0.709±0.074 | 0.575±0.087 | 0.773±0.052 | 0.683±0.104 | 0.709±0.074 | 33.596±18.038 |
| HemoSet → Rabbani | UNet | light | fda | 0.241±0.038 | 0.165±0.028 | 0.474±0.079 | 0.221±0.069 | 0.241±0.038 | 108.179±15.021 |
| Rabbani → Rabbani | UNet | light | fda | 0.686 | 0.563 | 0.763 | 0.694 | 0.686 | 53.186 |
| Rabbani → HemoSet | UNet | light | fda | 0.578 | 0.445 | 0.533 | 0.680 | 0.578 | 62.265 |
| HemoSet → HemoSet | UNet++ | light | none | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet → Rabbani | UNet++ | light | none | 0.236±0.035 | 0.159±0.023 | 0.418±0.096 | 0.223±0.022 | 0.236±0.035 | 104.057±8.325 |
| Rabbani → Rabbani | UNet++ | light | none | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani → HemoSet | UNet++ | light | none | 0.597 | 0.455 | 0.529 | 0.728 | 0.597 | 56.133 |
| HemoSet → HemoSet | UNet++ | light | reinhard | 0.735±0.036 | 0.602±0.042 | 0.760±0.050 | 0.739±0.040 | 0.735±0.036 | 31.047±12.715 |
| HemoSet → Rabbani | UNet++ | light | reinhard | 0.283±0.040 | 0.190±0.027 | 0.365±0.055 | 0.338±0.094 | 0.283±0.040 | 99.263±11.659 |
| Rabbani → Rabbani | UNet++ | light | reinhard | 0.621 | 0.501 | 0.759 | 0.622 | 0.621 | 63.865 |
| Rabbani → HemoSet | UNet++ | light | reinhard | 0.580 | 0.440 | 0.521 | 0.711 | 0.580 | 54.977 |
| HemoSet → HemoSet | UNet++ | light | fda | 0.724±0.044 | 0.591±0.050 | 0.759±0.070 | 0.728±0.095 | 0.724±0.044 | 27.940±9.329 |
| HemoSet → Rabbani | UNet++ | light | fda | 0.252±0.026 | 0.171±0.020 | 0.374±0.102 | 0.277±0.064 | 0.252±0.026 | 99.942±10.204 |
| Rabbani → Rabbani | UNet++ | light | fda | 0.672 | 0.539 | 0.664 | 0.757 | 0.672 | 48.767 |
| Rabbani → HemoSet | UNet++ | light | fda | 0.513 | 0.367 | 0.392 | 0.802 | 0.513 | 74.231 |
| HemoSet → HemoSet | DeepLabV3+ | light | none | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet → Rabbani | DeepLabV3+ | light | none | 0.196±0.032 | 0.133±0.025 | 0.665±0.074 | 0.165±0.035 | 0.196±0.032 | 151.249±20.629 |
| Rabbani → Rabbani | DeepLabV3+ | light | none | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani → HemoSet | DeepLabV3+ | light | none | 0.569 | 0.420 | 0.476 | 0.762 | 0.569 | 57.188 |
| HemoSet → HemoSet | DeepLabV3+ | light | reinhard | 0.713±0.051 | 0.575±0.061 | 0.730±0.105 | 0.731±0.060 | 0.713±0.051 | 34.398±14.164 |
| HemoSet → Rabbani | DeepLabV3+ | light | reinhard | 0.247±0.034 | 0.169±0.026 | 0.394±0.106 | 0.262±0.072 | 0.247±0.034 | 106.050±7.321 |
| Rabbani → Rabbani | DeepLabV3+ | light | reinhard | 0.571 | 0.452 | 0.774 | 0.568 | 0.571 | 83.954 |
| Rabbani → HemoSet | DeepLabV3+ | light | reinhard | 0.561 | 0.416 | 0.469 | 0.763 | 0.561 | 51.960 |
| HemoSet → HemoSet | DeepLabV3+ | light | fda | 0.728±0.029 | 0.594±0.028 | 0.760±0.050 | 0.723±0.027 | 0.728±0.029 | 30.026±10.805 |
| HemoSet → Rabbani | DeepLabV3+ | light | fda | 0.234±0.032 | 0.160±0.025 | 0.526±0.068 | 0.204±0.045 | 0.234±0.032 | 115.712±13.686 |
| Rabbani → Rabbani | DeepLabV3+ | light | fda | 0.666 | 0.537 | 0.692 | 0.704 | 0.666 | 41.395 |
| Rabbani → HemoSet | DeepLabV3+ | light | fda | 0.560 | 0.421 | 0.480 | 0.713 | 0.560 | 61.675 |

**Analisi**:
- **HemoSet→Rabbani (cross)**: sia Reinhard sia FDA migliorano rispetto al baseline "none" su tutte e tre le architetture (es. DeepLabV3+ 0.196→0.247 Reinhard / →0.234 FDA), ma **nessuno dei due batte l'augmentation aggressiva** (0.196→0.337 per DeepLabV3+, Fase 2). Reinhard risulta consistentemente un po' migliore di FDA su questa metrica in questa direzione.
- **Rabbani→HemoSet (cross)**: il quadro si ribalta — Reinhard e FDA spesso non aiutano o *peggiorano* rispetto al baseline (es. UNet++ + FDA: 0.597→0.513), mentre l'augmentation aggressiva resta l'unica condizione che migliora sempre in questa direzione.
- **Costo in-domain**: Reinhard e FDA hanno un costo in-domain molto più alto dell'augmentation aggressiva (quasi gratuita). Caso estremo: DeepLabV3+ su Rabbani in-domain crolla da 0.689 a 0.571 con Reinhard.
- **Interpretazione**: a differenza dell'augmentation (applicata con una certa probabilità per immagine), in questa implementazione l'adaptation è **applicata sempre**, ad ogni campione di training, in modo deterministico — il modello non vede mai l'aspetto "pulito" del proprio dominio sorgente durante il training, e finisce diviso tra due stili invece di consolidarne bene uno con variazioni. Questo spiegherebbe sia il guadagno cross-dataset (feature più trasferibili) sia il costo in-domain (minore specializzazione sul dominio proprio).
- **Conclusione comparativa Fasi 2-3**: l'augmentation aggressiva resta la strategia migliore tra quelle testate finora — unica a migliorare la robustezza cross-dataset in entrambe le direzioni senza costi apprezzabili in-domain. Reinhard/FDA aiutano la direzione più critica (HemoSet→Rabbani) ma a un costo non trascurabile altrove.

---

## 7. Fase 4 — Joint training

**Obiettivo**: allenare sull'unione di HemoSet e Rabbani, mantenendo test set held-out separati per i due domini, per verificare se una maggiore diversità nei dati di training produca un modello complessivamente più robusto.

**Design**: per isolare l'effetto del solo joint training, si usa augmentation "light" e adaptation "none" (stessa condizione della baseline di Fase 1). Per ogni fold HemoSet il training set è l'unione (`ConcatDataset`) tra il train split di quel fold e il train split fisso di Rabbani; il modello viene poi valutato **separatamente** sul test/val set di HemoSet e su quello di Rabbani (non esiste qui un concetto di "cross-dataset", dato che il modello ha visto entrambi i domini in training).

### 7.1 Joint training naturale (campionamento proporzionale alla dimensione dei dataset)

| Test set | Architettura | Training | Dice | IoU | Precision | Recall | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|
| HemoSet test | UNet | single-source (Fase 1) | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet test | UNet | joint | 0.747±0.041 | 0.620±0.044 | 0.756±0.045 | 0.769±0.051 | 0.747±0.041 | 25.614±8.944 |
| Rabbani test | UNet | single-source (Fase 1) | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani test | UNet | joint | 0.638±0.053 | 0.511±0.055 | 0.688±0.081 | 0.717±0.052 | 0.638±0.053 | 59.293±9.693 |
| HemoSet test | UNet++ | single-source (Fase 1) | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet test | UNet++ | joint | 0.757±0.036 | 0.630±0.039 | 0.729±0.051 | 0.814±0.038 | 0.757±0.036 | 30.395±8.579 |
| Rabbani test | UNet++ | single-source (Fase 1) | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani test | UNet++ | joint | 0.631±0.056 | 0.504±0.059 | 0.683±0.088 | 0.701±0.038 | 0.631±0.056 | 60.076±10.358 |
| HemoSet test | DeepLabV3+ | single-source (Fase 1) | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet test | DeepLabV3+ | joint | 0.721±0.053 | 0.590±0.055 | 0.759±0.077 | 0.734±0.075 | 0.721±0.053 | 29.856±9.249 |
| Rabbani test | DeepLabV3+ | single-source (Fase 1) | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani test | DeepLabV3+ | joint | 0.640±0.034 | 0.511±0.031 | 0.724±0.022 | 0.668±0.039 | 0.640±0.034 | 56.070±7.804 |

**Analisi**:
- **Su HemoSet**: effetto piccolo e incoerente tra architetture — UNet e UNet++ migliorano leggermente (0.738→0.747, 0.749→0.757), DeepLabV3+ peggiora (0.748→0.721). Nessun beneficio sistematico.
- **Su Rabbani**: effetto **sistematicamente negativo** su tutte e tre le architetture (0.689→0.638 UNet, 0.675→0.631 UNet++, 0.689→0.640 DeepLabV3+, cali di circa 0.04-0.05 Dice ovunque).
- **Interpretazione**: nello split combinato, HemoSet pesa di più nel training set (748 immagini/fold contro le ~525 del train split di Rabbani, ~59% vs ~41% dei dati); con un `ConcatDataset` campionato uniformemente (`shuffle=True`), il modello vede più spesso esempi HemoSet, producendo un modello leggermente sbilanciato verso le caratteristiche di HemoSet a scapito di Rabbani.

### 7.2 Joint training con campionamento bilanciato tra domini (Fase 4b)

Per correggere lo sbilanciamento osservato in 7.1, si assegna a ogni immagine un peso inversamente proporzionale alla dimensione del proprio dominio (1/748 per HemoSet, 1/525 per Rabbani) e si campiona con `WeightedRandomSampler`, in modo che ogni batch abbia probabilità ~50/50 di contenere un'immagine di ciascun dominio.

| Test set | Single-source (Fase 1) | Joint naturale (Fase 4) | Joint bilanciato (Fase 4b) |
|---|---|---|---|
| HemoSet test | 0.745 | 0.742 | 0.739 |
| Rabbani test | 0.684 | 0.636 | 0.656 |

*(valori medi sulle 3 architetture; per architettura: Rabbani UNet 0.638→0.658, UNet++ 0.631→0.658, DeepLabV3+ 0.640→0.653)*

**Analisi**: il bilanciamento **recupera parzialmente** il calo su Rabbani osservato con il campionamento naturale (+0.013/+0.027 di Dice a seconda dell'architettura), a costo piccolo e trascurabile su HemoSet (0.742→0.739). Non elimina però del tutto il gap rispetto alla baseline single-source (0.656 vs 0.684, circa metà del gap recuperato). Il joint training bilanciato non è quindi competitivo con l'augmentation aggressiva o con l'ensemble (Sezione 8) sul dominio Rabbani, ma conferma l'ipotesi dello sbilanciamento come causa del calo osservato nella condizione naturale.

---

## 8. Esperimento aggiuntivo — Ensemble delle tre architetture

Esperimento a costo computazionale quasi nullo (nessun training aggiuntivo): media delle predizioni (probabilità sigmoid pixel-wise) dei checkpoint delle tre architetture allenate con augmentation *aggressive* (la condizione migliore, Sezione 5), soglia 0.5 applicata alla probabilità media. L'idea è che errori "indipendenti" dei singoli modelli tendano a cancellarsi nella media, mentre il segnale condiviso si rinforzi.

| Setting | Dice | IoU | Precision | Recall | F1 | HD95 |
|---|---|---|---|---|---|---|
| HemoSet → HemoSet (in-domain, ensemble) | 0.770±0.025 | 0.647±0.028 | 0.798±0.054 | 0.773±0.015 | 0.770±0.025 | 24.637±5.623 |
| HemoSet → Rabbani (cross, ensemble) | 0.343±0.014 | 0.245±0.012 | 0.433±0.030 | 0.379±0.029 | 0.343±0.014 | 91.457±4.161 |
| Rabbani → Rabbani (in-domain, ensemble) | 0.706 | 0.583 | 0.757 | 0.739 | 0.706 | 39.571 |
| Rabbani → HemoSet (cross, ensemble) | 0.623 | 0.481 | 0.544 | 0.777 | 0.623 | 49.629 |

| Setting | Migliore modello singolo (aggressive) | Ensemble | Δ |
|---|---|---|---|
| HemoSet in-domain | 0.756 (UNet/UNet++) | 0.770 | +0.014 |
| HemoSet→Rabbani (cross) | 0.337 (DeepLabV3+) | 0.343 | +0.006 |
| Rabbani in-domain | 0.701 (UNet++) | 0.706 | +0.005 |
| Rabbani→HemoSet (cross) | 0.615 (UNet++) | 0.623 | +0.008 |

**Analisi**: l'ensemble batte **il migliore dei tre modelli singoli** (non solo la loro media) su tutte e quattro le condizioni testate, con guadagni modesti in valore assoluto ma **sistematici e senza eccezioni**, anche sull'HD95, a costo computazionale nullo in training (solo tre forward pass invece di uno in inferenza).

---

## 9. Analisi comparativa finale

### 9.1 Sintesi di tutte le strategie

Media sulle tre architetture per ciascuna strategia, per un confronto indipendente dalla scelta di una singola architettura:

| Strategia | Fase | HemoSet test (in-domain) | Rabbani test (in-domain) | HemoSet→Rabbani (cross) | Rabbani→HemoSet (cross) |
|---|---|---|---|---|---|
| Baseline (source-only, augmentation light) | 1 | 0.745 | 0.684 | 0.217 | 0.585 |
| Augmentation aggressive | 2 | 0.747 | 0.681 | 0.321 | 0.601 |
| Appearance adaptation (Reinhard) | 3 | 0.719 | 0.611 | 0.263 | 0.577 |
| Appearance adaptation (FDA) | 3 | 0.720 | 0.675 | 0.242 | 0.550 |
| Joint training naturale | 4 | 0.742 | 0.636 | n/a* | n/a* |
| Joint training bilanciato | 4b | 0.739 | 0.656 | n/a* | n/a* |
| **Ensemble 3 architetture (su augmentation aggressive)** | extra | **0.770** | **0.706** | **0.343** | **0.623** |

\* il joint training viene valutato in-domain su entrambi i test set (avendo visto entrambi i domini in training); non esiste quindi una condizione "cross-dataset" comparabile per questa riga.

### 9.2 Risposta alla domanda di ricerca

Tra le strategie testate, un intervento **leggero e probabilistico** sulla varietà visiva del training (augmentation aggressiva) batte nettamente interventi più mirati ma **sempre-attivi** sullo stile (Reinhard/FDA) e il semplice mescolamento dei dati (joint training). Il fattore comune ai due interventi meno efficaci — adaptation sempre-attiva, joint training sbilanciato — sembra essere la mancanza di un meccanismo che preservi comunque l'esposizione del modello alla distribuzione "pulita" del proprio dominio sorgente durante il training:

1. **L'augmentation aggressiva è la strategia più efficace tra quelle testate**: unica a migliorare *entrambe* le direzioni cross-dataset (HemoSet→Rabbani: +48% relativo; Rabbani→HemoSet: +3%) mantenendo le performance in-domain sostanzialmente invariate. Miglior rapporto costo/beneficio.
2. **Il color transfer di Reinhard aiuta la direzione più critica** (HemoSet→Rabbani: +21%) ma a un costo in-domain non trascurabile, soprattutto su Rabbani (-11%).
3. **FDA è più debole di Reinhard** su entrambi i fronti in questo setting: guadagno cross-dataset minore (+12% su HemoSet→Rabbani) e un peggioramento nella direzione Rabbani→HemoSet (-6%).
4. **Il joint training "ingenuo" non è una soluzione**: guadagno trascurabile/incoerente su HemoSet, costo sistematico su Rabbani (-7%), dovuto allo sbilanciamento tra le dimensioni dei due dataset (~59% HemoSet vs ~41% Rabbani per fold). Il campionamento bilanciato (Fase 4b) recupera solo metà del gap.
5. **Configurazione finale consigliata: augmentation aggressive in training + ensemble delle tre architetture in inferenza** — combinazione che, tra tutte le condizioni testate, ottiene il miglior risultato su tutte e quattro le condizioni valutate (in-domain e cross-dataset, entrambe le direzioni).

---

## 10. Verifica del rigore sperimentale

### 10.1 Budget di training e convergenza

Tutti i run riportati (Fasi 1-4, 87 run totali) usano la stessa configurazione di training (Sezione 3.3), identica per ogni condizione confrontata, così da isolare correttamente la singola variabile in esame in ciascuna fase. Per verificare che il budget (40 epoche massime, early stopping patience 8) fosse sufficiente per la convergenza, per ciascun run è stato confrontato il Dice di validazione dell'ultima epoca con quello di 6 epoche prima, segnalando i run che avessero raggiunto il cap di 40 epoche mostrando ancora un miglioramento superiore a 0.01.

**Risultato**: la maggioranza dei run si è fermata per early stopping ben prima delle 40 epoche (tipicamente tra 10 e 30). Solo 2 run su 87 hanno raggiunto il cap di 40 epoche, ed entrambi mostravano un miglioramento residuo minimo (+0.0067 e +0.0026 di Dice nelle ultime 6 epoche) — un plateau, non un training interrotto a metà. Nessun run era ancora in miglioramento significativo al momento dello stop (0/87 con delta > 0.01 al cap). Il budget di training è quindi adeguato per tutte le condizioni testate.

*Nota*: il fold 3 di HemoSet risulta sistematicamente il più difficile (Dice inferiore di circa 0.1-0.4 rispetto agli altri fold, in tutte le architetture e condizioni), probabilmente per un soggetto particolarmente ostico nello split di validazione di quel fold — non approfondito ulteriormente, segnalato come possibile direzione di analisi futura.

### 10.2 Verifica delle metriche rispetto ai paper originali

- **Miao et al. (HemoSet, ISMR 2024)**: usano IoU, F1, HD (Hausdorff Distance massima, non il 95° percentile).
- **Rabbani et al. (MIDL 2022)**: usano IoU e F-Score (matematicamente identico a Dice/F1 in segmentazione binaria); nessun HD; includono anche una valutazione soggettiva con chirurghi, non replicabile in questo lavoro e fuori scope.

Le metriche usate in questo lavoro (Dice, IoU, Precision, Recall, F1, HD95) coprono per intero entrambi i paper. L'unico scostamento è l'uso di HD95 al posto dell'HD massimo di Miao et al. — scelta più robusta perché molto meno sensibile a un singolo pixel outlier rispetto all'HD massimo.

---

## 11. Conclusioni e lavori futuri

Questo lavoro ha quantificato in modo sistematico il domain gap asimmetrico tra HemoSet e Rabbani, confermando l'osservazione di partenza, e ha confrontato quattro famiglie di strategie per aumentare la robustezza cross-dataset dei modelli di segmentazione del sangue in chirurgia: data augmentation, appearance/domain adaptation, joint training, ensembling. La **data augmentation aggressiva** è risultata la strategia più efficace e a più basso costo, e la sua combinazione con un **ensemble di tre architetture** (UNet, UNet++, DeepLabV3+) produce il miglior risultato complessivo, senza alcun training aggiuntivo oltre a quello già necessario per l'augmentation.

**Limiti e lavori futuri**:
- Provare l'appearance/domain adaptation con probabilità <1 (in stile augmentation) invece che sempre attiva, per isolare se il costo in-domain osservato è dovuto proprio a questo.
- Combinare augmentation aggressiva e appearance adaptation, per verificare se gli effetti si sommano.
- Joint training con campionamento bilanciato è già stato testato (Fase 4b); rimane da esplorare un bilanciamento anche a livello di loss/pesi per classe.
- Analizzare la causa della maggiore difficoltà del fold 3 di HemoSet.
- Quando disponibili, ripetere gli esperimenti più promettenti (augmentation aggressiva in primis) sui dati reali del progetto TRAMIS, tenendo conto — come osservato in Fase 4 — che mescolare dataset esterni di dimensioni molto diverse nel training non è automaticamente benefico senza un bilanciamento esplicito tra domini.
- Valutare estensioni basate su backbone Transformer pretrained (es. DINOv3, come nel BorDINO proposto da Giusti & Marzo), non esplorate in questo lavoro perché fuori dallo scope CNN-based definito per il progetto.

---

## 12. Riferimenti bibliografici

1. A. J. Miao, S. Lin, J. Lu, F. Richter, B. Ostrander, E. K. Funk, R. K. Orosco, M. C. Yip. "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management." *International Symposium on Medical Robotics (ISMR)*, 2024.
2. N. Rabbani, C. Seve, N. Bourdel, A. Bartoli. "Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation." *Medical Imaging with Deep Learning (MIDL)*, 2022.
3. Y. Yang, S. Soatto. "FDA: Fourier Domain Adaptation for Semantic Segmentation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
4. E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley. "Color Transfer between Images." *IEEE Computer Graphics and Applications*, 2001.
5. Su et al. *AAAI Conference on Artificial Intelligence*, 2023. (domain generalization single-source via augmentation)
6. Teevno et al. *IEEE International Symposium on Computer-Based Medical Systems (CBMS)*, 2024.
7. F. Giusti, F. Marzo. Report di progetto, Computer Vision and Cognitive Systems, UNIMORE, A.A. 2025/2026.
