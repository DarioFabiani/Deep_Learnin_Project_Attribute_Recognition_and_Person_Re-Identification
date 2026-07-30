# Deep Learning Course Assignment 20/21
### Attribute recognition & person re-identification on Market-1501

Two tasks on 64×128 pedestrian crops:

1. **Attribute recognition** — predict 29 attributes for each image in `test/`.
2. **Person re-identification** — for each image in `queries/`, rank the images of
   `test/` by identity similarity (evaluated as mAP over the top-20).

## Notebooks

| notebook | cos'è |
|---|---|
| `Quadruplet_Network_for_attribute_recognition_and_person_re-ID.ipynb` | il notebook originale, **corretto**. Stessa architettura, stessa struttura, stesse loss; sono stati risolti i bug che rendevano i risultati privi di significato. Ogni correzione è marcata con un commento `# FIX:`. |
| `Quadruplet_Network_v2_improved.ipynb` | riscrittura con la ricetta moderna: PK sampler + hard mining, BNNeck, testa ID, GeM, AdamW + cosine, AMP, metriche complete (mAP, CMC, F1 per attributo). |

`FIXES.md` documenta tutti i bug trovati, il loro impatto e le note hardware.

## Come eseguirli

### In locale (basta una GPU da 8 GB)

```bash
pip install torch torchvision pandas scikit-learn matplotlib tqdm tensorboard jupyter
unzip dataset.zip -d data
jupyter lab
```

I notebook rilevano automaticamente il percorso dei dati (`./data` in locale,
`/content/data` su Colab); per sovrascriverlo:

```bash
DATA_ROOT=/percorso/di/data jupyter lab
```

Non c'è più alcuna dipendenza da `google.colab`: le celle specifiche di Colab
(mount di Drive, `files.download`) sono attive solo quando il notebook gira lì.

### Su Colab

1. Carica il notebook e `dataset.zip` nella stessa cartella (oppure metti lo zip in
   `MyDrive/DEEP_LEARNING_PROJECT/`).
2. Runtime → Change runtime type → GPU.
3. Esegui tutte le celle.

## Requisiti hardware

Il preset di default della v2 (ResNet50, `last_stride=1`, 64 immagini per step,
256×128, AMP) usa circa **5.2 GB di VRAM** e gira in ~2–3 min/epoca su una RTX
4070 Laptop. Per GPU più piccole:

```python
cfg = Config(backbone='resnet18', last_stride=2, p_identities=24)   # ~1.9 GB
```

La cella `measure_peak_memory(cfg)` misura il picco reale sulla tua GPU prima di
lanciare il training. Dettagli e tabella completa in `FIXES.md`, sezione 6.

## Output prodotti

- `classification_test.csv` — 19.679 righe × 29 attributi, indicizzate per nome file.
- `reid_test.txt` — una riga per query: `query.jpg: top1.jpg, top2.jpg, ...` (top-20).

## Autori

Giacomo Lazzerini · Dario Fabiani — University of Trento
