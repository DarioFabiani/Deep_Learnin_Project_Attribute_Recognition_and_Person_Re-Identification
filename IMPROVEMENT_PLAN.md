# Piano Completo — Quadruplet Network: Bug Fix + Notebook Migliorato

## Context

Progetto universitario (Università di Trento, Giacomo Lazzerini & Dario Fabiani) che implementa
una Quadruplet Network per:
1. **Attribute Recognition** — 29 attributi di pedoni (binari + multiclasse)
2. **Person Re-ID** — cosine similarity su embedding, valutato con mAP@20

Tutto il codice è in un unico notebook Jupyter:
`Quadruplet_Network_for_attribute_recognition_and_person_re-ID.ipynb`

L'obiettivo è: correggere il notebook originale come baseline (Fase 0), poi creare
`notebook_improved.ipynb` con i miglioramenti dalla ricerca (Fase 2).

---

## Architettura Attuale (Baseline)

```
ResNet18 pretrained (512-dim) → GlobalAvgPool
  ├── ClassificationBlock × 29  (FC 512→256 + BN + LeakyReLU + Dropout(0.5) + FC 256→k)
  │     → AttributeLoss = BCEWithLogitsLoss (binari) + CrossEntropyLoss (multiclasse)
  └── embedding 512-dim
        → QuadrupletLoss (margin1=2.0, margin2=1.0, distanza L2 quadratica)
        → cosine similarity per Re-ID

Loss totale = 0.8 × AttributeLoss + 0.2 × QuadrupletLoss
Optimizer: SGD lr=0.001, 50 epoch, batch=48, negative mining RANDOM
Dataset: Market-1501 (64×128px, 751 ID train, 12.989 immagini)
```

---

## FASE 0 — Bug da correggere nel notebook originale

> Queste correzioni sono già parzialmente implementate in `quadruplet_network.py` (branch `dev`),
> ma il file non include tutti i bug nuovi trovati da Opus 5.

### BUG-0 [CRITICO — non era nella review iniziale] Disallineamento immagine↔etichette
**Cella 21, classe `MarketDataset.__init__`**
```python
# ERRATO: usa os.listdir (12989 file, ordine arbitrario)
self.list_dir = sorted(os.listdir(self.root_dir))
# __getitem__ legge immagine da list_dir[index] ma label da annotations.iat[index]
# → zero corrispondenze corrette su 9348 campioni
```
**Fix:** costruire `self.files` dal dataframe, non dalla directory:
```python
self.files = self.df['image_name'].tolist()   # ordine coerente con le label
```
Per i dataset senza annotazioni (test/query) usare una classe separata con `sorted(os.listdir())`.

**Fonte:** Opus 5 synthesis — analisi diretta del codice.

---

### BUG-1 [CRITICO] Salvataggio modello non addestrato
**Cella 50, funzione `main()`**
```python
# ERRATO:
model = Backbone(attribute_map=att_map)   # nuovo modello, pesi random
torch.save(model.state_dict(), 'model')  # salva pesi random, non 'net'
```
**Fix:** `torch.save(net.state_dict(), 'best_model.pth')`

**Fonte:** review iniziale (finding #2).

---

### BUG-2 [CRITICO] `RandomCrop(32)` dopo `Resize(224,224)` — ricetta CIFAR
**Cella 24, `train_tfms`**
```python
T.Resize(size=(224, 224)),
T.RandomCrop(32, padding=4)   # ← ritaglia 32×32 da 224×224: il 98% del pedone è buttato
```
**Fix:** risoluzione standard Re-ID e crop proporzionale:
```python
T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
T.Pad(10),
T.RandomCrop((256, 128)),
```

**Fonte:** Opus 5 synthesis — Finding 5 (data augmentation) + analisi codice.

---

### BUG-3 [CRITICO] mAP calcolata su dati shufflati → priva di significato
**Cella 28, `validation_test_loader`**
```python
shuffle=True, drop_last=True   # ← distrugge corrispondenza indice↔immagine
```
**Fix:** `shuffle=False, drop_last=False` su TUTTI i loader di feature extraction.

**Fonte:** review iniziale (finding #4) + Opus 5.

---

### BUG-4 [ALTO] `F.dropout` non rispetta `net.eval()`
**Cella 32, `ClassificationBlock.forward`**
```python
x = F.dropout(x, p=0.5)   # training=True di default: dropout attivo anche in valutazione
```
**Fix:** `self.drop = nn.Dropout(0.5)` come modulo, oppure `F.dropout(x, 0.5, self.training)`.

**Fonte:** Opus 5 synthesis.

---

### BUG-5 [ALTO] Argomenti SGD invertiti → momentum e weight_decay sbagliati
**Cella 40, `get_optimizer`**
```python
def get_optimizer(net, lr, wd, momentum):
    return torch.optim.SGD(net.parameters(), lr, momentum)
    # 3° posizionale di SGD = momentum → riceve wd=1e-6 come momentum
    # momentum=0.5 è ignorato, weight_decay=0
```
**Fix:** argomenti keyword espliciti:
```python
torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
```

**Fonte:** Opus 5 synthesis.

---

### BUG-6 [ALTO] `AttributesLoss` somma 29 loss → schiaccia la quadruplet loss di ~30×
**Cella 37, `AttributesLoss.forward`**
Con λ=0.8: `0.8 × (29 × 0.6) ≈ 14` contro `0.2 × quadruplet ≈ 0.2`.
Il segnale di metric learning è numericamente irrilevante.
**Fix:** media sulle teste invece di somma:
```python
return (binary_losses + cross_losses) / len(preds)
```

**Fonte:** Opus 5 synthesis.

---

### BUG-7 [ALTO] Test set usa `train_tfms` (augmentazioni random)
**Cella 26**
```python
test_ds = MarketDataset(..., transform=train_tfms, ...)   # ← sbagliato
queries_ds = MarketDataset(..., transform=train_tfms, ...) # ← sbagliato
```
**Fix:** `transform=valid_tfms` per tutti i dataset non di training.

**Fonte:** review iniziale (finding #3).

---

### BUG-8 [MEDIO] `predicted[i][1]` in `train()` — accuracy errata
**Cella 44**
```python
accuracy[i] += predicted[i][1].eq(...)   # [1] prende uno scalare, non il tensore batch
```
**Fix:** `predicted[i].eq(...)`

**Fonte:** review iniziale (finding #1).

---

### BUG-9 [MEDIO] CSV submission con indici non allineati
**Cella 60**
`index=list(os.listdir(config['test_path']))` (non ordinato) mentre il dataset itera
`sorted(os.listdir())`. I nomi file sono disallineati.
**Fix:** far restituire il filename dal dataset e usarlo come indice.

---

### BUG-10 [BASSO] Ogni testa invocata 3 volte nel forward — 3× FLOPs
**Cella 35, `Backbone.forward`** — `prob_pred_label` chiama `self.__getattr__(...)` 3 volte per
ogni attributo. Fix: calcolare i logit una volta sola e derivare le probabilità.

---

## FASE 2 — Miglioramenti per il secondo notebook

Ordine di implementazione: impatto × facilità. Ogni step è isolato → ablation table.

### F2-1 · BNNeck + ID Classification Loss + Label Smoothing ⭐ PRIORITÀ MASSIMA

**Cosa fa:** crea due spazi di feature separati:
- `f_t` (pre-BN) → quadruplet loss (spread euclideo)
- `f_i` (post-BN) → ID loss, attributi, retrieval

**Impatto atteso:** +12-14 mAP (ablation Luo et al.: 69.22 → 83.43 mAP).

**Fonti:**
- *"Bag of Tricks and a Strong Baseline for Deep Person Re-ID"*, Luo et al., CVPR-W 2019, arXiv:1903.07071
- Repo reference: `michuanhaohao/reid-strong-baseline`
- Research findings: #6, #8, #17, #20

**Codice chiave:**
```python
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1./self.p)

class Backbone(nn.Module):
    def __init__(self, attribute_map, num_ids):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.layer4[0].conv1.stride = (1,1)          # last-stride=1: +1-2 mAP
        base.layer4[0].downsample[0].stride = (1,1)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = GeM()
        self.bnneck = nn.BatchNorm1d(512)
        self.bnneck.bias.requires_grad_(False)
        self.id_head = nn.Linear(512, num_ids, bias=False)   # num_ids = n. ID nel train split
        # ... teste attributi come prima, ma su f_i

id_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```
**Nota:** gli ID vanno rimappati 0..N-1 sul solo train split (~540 ID dopo lo split 72/28).

---

### F2-2 · P×K Sampler + Batch-Hard Quadruplet ⭐ PRIORITÀ ALTA

**Cosa fa:** sostituisce il random sampling con campionamento strutturato P identità × K
immagini, poi seleziona la coppia anchor-positivo più difficile e il negativo più vicino.

**Impatto atteso:** mAP 52.22% → 66.59% (+14.37%), R1 +5.14% (verificato empiricamente).

**Fonti:**
- *"In Defense of the Triplet Loss for Person Re-Identification"*, Hermans et al., 2017, arXiv:1703.07737
- Research finding: #10
- **Correzione Opus 5:** usare P=12, K=4 (non P=4, K=12 come suggerito dall'agente Haiku) →
  12 identità negative disponibili per ogni ancora invece di 3.

**Parametri margini:** passando da distanza L2-quadratica a L2 standard, riscalare i margini
da (2.0, 1.0) a **(0.3, 0.15)**. Lasciare 2.0 con distanza non-quadrata satura i gradienti.

```python
class RandomIdentitySampler(Sampler):
    def __init__(self, pids, P=12, K=4): ...

class BatchHardQuadruplet(nn.Module):
    def __init__(self, m1=0.3, m2=0.15): ...
    def forward(self, f, pids):
        d = euclidean_dist(f, f)
        d_ap = hardest_positive(d, pids)
        d_an, n1_idx = hardest_negative(d, pids)
        d_nn = second_hardest_negative(d, pids, n1_idx)
        return (relu(d_ap - d_an + m1) + relu(d_ap - d_nn + m2)).mean()
```

---

### F2-3 · LR Warm-up + Cosine Annealing + Ottimizzatore corretto

**Cosa fa:** warm-up lineare per 10 epoch → cosine decay fino a epoch 60.
Obbligatorio quando si aggiunge la ID head (pesi random → gradienti esplosivi nelle prime epoch).

**Impatto atteso:** +2-5% accuratezza generale.

**Fonte:** Research finding #7 · *"Bag of Tricks"* (Luo et al.)

```python
optimizer = torch.optim.Adam(net.parameters(), lr=3.5e-4, weight_decay=5e-4)
WARMUP, TOTAL = 10, 60
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lambda e: (e+1)/WARMUP if e < WARMUP else 0.5*(1+math.cos(math.pi*(e-WARMUP)/(TOTAL-WARMUP))))
```

---

### F2-4 · Random Erasing

**Cosa fa:** cancella una regione rettangolare casuale dell'immagine durante il training.
Zero costo architetturale, 1 riga di codice aggiunta dopo `Normalize`.

**Impatto atteso:** +2.93% R1, +2-5 mAP su Market-1501.

**Fonte:** *"Random Erasing Data Augmentation"*, Zhong et al., 2020, arXiv:2002.11371 ·
Research finding #5, #15

```python
T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
# da aggiungere DOPO T.Normalize() in train_tfms
```

---

### F2-5 · `pos_weight` per BCE + Metrica mA (mean Accuracy bilanciata)

**Cosa fa:** corregge il class imbalance sugli attributi binari (66% degli attributi ha
frequenza <10%). La semplice accuracy maschera il problema: predire sempre 0 su `upyellow`
dà 99%+ di accuracy ma è inutile.

**Impatto atteso:** alto sul task attributi; cambia la narrativa del report.

**Fonte:** Research finding #2, #18 · *"VLM-PAR"*, arXiv:2512.22217

```python
pos_count = train_df[binary_cols].values.sum(0)
neg_count = len(train_df) - pos_count
pos_weight = torch.tensor(neg_count / np.maximum(pos_count, 1)).clamp(max=20.).to(device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight[idx])

# Metrica corretta:
mA = np.mean([0.5*(TP[i]/max(P[i],1) + TN[i]/max(N[i],1)) for i in range(n_attr)])
```

---

### F2-6 · Best-Checkpoint + Early Stopping + L2-Norm per retrieval

**Cosa fa:** salva il modello migliore su mAP di validazione (non l'ultimo epoch).
In retrieval usa feature post-BNNeck L2-normalizzate.

**Fonte:** Opus 5 synthesis.

```python
best_mAP, patience = 0., 0
for e in range(TOTAL):
    train_one_epoch(...); scheduler.step()
    if e >= 10 and e % 2 == 0:
        mAP = evaluate(...)
        if mAP > best_mAP:
            best_mAP, patience = mAP, 0
            torch.save(net.state_dict(), 'best.pth')
        else:
            patience += 1
            if patience >= 8: break
net.load_state_dict(torch.load('best.pth'))

# Retrieval vettorializzato (no loop per query):
feats = F.normalize(net(x), dim=1)
sims  = feats_q @ feats_g.t()
```

---

### F2-7 · Flip Test-Time Augmentation

**Cosa fa:** media le feature dell'immagine originale e della sua versione flippata orizzontalmente.
Non citato dalla ricerca dei 20 agenti ma standard in Re-ID.

**Impatto atteso:** +0.5-1 mAP.

```python
f = net(x) + net(torch.flip(x, dims=[3]))
f = F.normalize(f, dim=1)
```

---

### F2-8 · k-Reciprocal Re-Ranking ⚠️ LACUNA DELLA RICERCA

**Cosa fa:** post-processing sulla matrice di distanza già calcolata, zero training aggiuntivo.
**Non citato da nessuno dei 20 agenti** ma è il miglioramento post-hoc più forte in Re-ID.

**Impatto atteso:** +8-10 mAP su Market-1501 (ogni riga della leaderboard ha una variante "(RK)").

**Fonte:** *"Re-ranking Person Re-identification with k-reciprocal Encoding"*, Zhong et al.,
CVPR 2017, arXiv:1701.08398 · Implementazione: `zhunzhong07/person-reid-triplet-loss-baseline`
(file `re_ranking.py`)

```python
from re_ranking import re_ranking
dist = re_ranking(q_feats, g_feats, k1=20, k2=6, lambda_value=0.3)
```
**Attenzione:** con mAP@20 su gallery piccola il guadagno è minore che sul protocollo
full-gallery Market-1501. Va dichiarato nel report.

---

### F2-9 · GeM Pooling — già incluso in F2-1

**Fonte:** *"Fine-tuning CNN Image Retrieval with No Human Annotation"*, Radenovic et al.,
TPAMI 2018, arXiv:1711.02512 · Research finding #13
Sostituisce `nn.AdaptiveAvgPool2d((1,1))` con il modulo `GeM` mostrato in F2-1.

---

### F2-10 · Mixed Precision (AMP) — abilitante per Colab

**Cosa fa:** ~2× velocità, ~40% memoria in meno → permette 256×128 input + più epoch.
Non citato dalla ricerca ma necessario per rendere fattibile tutto il resto su Colab T4.

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

---

### F2-11 · Circle Loss (alternativa/confronto al Quadruplet)

**Cosa fa:** pesi self-paced per ogni similarity score. Da usare come variante isolata
nella tabella di ablation (non sostituzione della baseline), per confronto diretto.

**Impatto atteso:** +2-4 mAP vs triplet standard.

**Fonte:** *"Circle Loss"*, Sun et al., CVPR 2020, arXiv:2002.10857 · Research finding #3, #16
Lib: `pytorch-metric-learning` → `from pytorch_metric_learning.losses import CircleLoss`
Parametri: `m=0.25, gamma=64`, applicato su `f_t` (pre-BNNeck).

---

### F2-12 · IBN-Net ResNet18 (cross-domain)

**Cosa fa:** drop-in replacement per ResNet18 che mescola Instance Norm (layer early,
invarianza appearance) e Batch Norm (layer late, discriminazione ID).

**Fonte:** *"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"*,
Pan et al., ECCV 2018 · Research finding #19

```python
import timm
base = timm.create_model('resnet18_ibn_a', pretrained=True, num_classes=0, global_pool='')
```
Utile per cross-camera generalization. Su Market-1501 single-domain il guadagno è modesto.

---

## Miglioramenti sconsigliati (con motivazione)

| Idea | Motivo |
|---|---|
| ViT / TransReID | 87.5M param, richiede 256×128, ~120 epoch → fuori budget Colab |
| ConvNeXt / EfficientNet | Stesso mAP o peggio di ResNet50 (controllato empirically) |
| PCB / MGN | Ha senso solo dopo che ID loss funziona; alta complessità |
| CLIP / SigLIP2 | Cambia natura al progetto (frozen VLM ≠ quadruplet network) |

---

## Nota sul protocollo di valutazione

La **"mAP@20"** del progetto NON è comparabile con i numeri Market-1501 SOTA perché:
1. Gallery è un sottoinsieme del train (non la gallery completa di 19.732 immagini)
2. Manca l'esclusione delle immagini same-camera/same-ID (junk images)
3. Troncato a rank 20

Il report deve dichiararlo esplicitamente prima di qualsiasi confronto con i 89.9% mAP citati
dalla letteratura.

Inoltre: le label degli attributi sono annotate **per identità**, non per singola immagine.
Tutte le immagini dello stesso ID condividono le stesse 29 label. Questo rende il task
attributi più facile del PAR standard (PA-100K, PETA) e va dichiarato.

---

## Tabella di Ablation Consigliata per il Report

| # | Configurazione | mAP@20 | Attr. mA |
|---|---|---|---|
| 0 | Notebook originale (con bug) | ~rumore | ~non interpretabile |
| 1 | **+ Fase 0: fix tutti i bug** ← vera baseline | | |
| 2 | + input 256×128 + warm-up/cosine + Adam | | |
| 3 | + **BNNeck + ID loss + label smoothing** | | |
| 4 | + **P×K sampler + batch-hard quadruplet** | | |
| 5 | + Random Erasing | | |
| 6 | + GeM + last-stride=1 | | |
| 7 | + pos_weight BCE | | |
| 8 | + flip TTA | | |
| 9 | + **k-reciprocal re-ranking** | | |
| 10 | (variante) Circle Loss al posto del quadruplet | | |
| 11 | (variante) IBN-a ResNet18 | | |

I salti attesi più grandi: righe 3 e 4. Se non si osservano, c'è ancora un bug.

---

## File da creare / modificare

| File | Azione |
|---|---|
| `Quadruplet_Network_for_attribute_recognition_and_person_re-ID.ipynb` | Correggere i bug Fase 0 (celle 21, 24, 26, 28, 32, 35, 37, 40, 44, 50, 60) |
| `quadruplet_network.py` (branch `dev`) | Aggiornare con i bug-fix mancanti (BUG-0, BUG-2, BUG-4, BUG-5, BUG-6, BUG-9) |
| `notebook_improved.ipynb` | Nuovo notebook con i miglioramenti Fase 2 |
| `re_ranking.py` | Copiare da `zhunzhong07/person-reid-triplet-loss-baseline` |

---

## Fonti / Paper di Riferimento

| Paper | Link | Finding |
|---|---|---|
| *Bag of Tricks and a Strong Baseline for Deep Person Re-ID* — Luo et al., CVPR-W 2019 | arXiv:1903.07071 | BNNeck, warm-up, label smoothing, last-stride=1 |
| *In Defense of the Triplet Loss for Person Re-ID* — Hermans et al., 2017 | arXiv:1703.07737 | Batch-hard mining, P×K sampler |
| *Random Erasing Data Augmentation* — Zhong et al., 2020 | arXiv:2002.11371 | Random Erasing |
| *Re-ranking with k-reciprocal Encoding* — Zhong et al., CVPR 2017 | arXiv:1701.08398 | k-reciprocal re-ranking |
| *Circle Loss* — Sun et al., CVPR 2020 | arXiv:2002.10857 | Circle Loss |
| *Fine-tuning CNN Image Retrieval* — Radenovic et al., TPAMI 2018 | arXiv:1711.02512 | GeM pooling |
| *IBN-Net* — Pan et al., ECCV 2018 | — | IBN-a backbone |
| *TransReID* — He et al., ICCV 2021 | arXiv:2102.04378 | ViT per Re-ID (solo riferimento) |
| *AGW baseline* — Ye et al., TPAMI 2021 | arXiv:2001.04193 | Survey + GeM in Re-ID |

**Repo di riferimento:**
- `michuanhaohao/reid-strong-baseline` — Bag of Tricks, BNNeck, ablation table
- `layumi/Person_reID_baseline_pytorch` — baseline solida, confronto backbone
- `zhunzhong07/Random-Erasing` — implementazione RE con hyperparameter ufficiali
- `zhunzhong07/person-reid-triplet-loss-baseline` — re_ranking.py

---

## Verifica

1. Eseguire il notebook corretto su Colab con GPU T4
2. Controllare che `accuracy[i]` in `train()` mostri valori sensati (non 0% o 100% fissi)
3. Verificare che la mAP@20 prima del training sia ~0.02-0.05 (random embedding), non più alta
4. Dopo il training: mAP dovrebbe essere >40% (vera baseline, post fix)
5. Dopo F2-1+F2-2+F2-3: mAP dovrebbe essere >60%
6. Compilare la tabella di ablation aggiungendo una riga per ogni step
