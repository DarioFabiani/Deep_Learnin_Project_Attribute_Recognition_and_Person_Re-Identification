# Bug report e note di implementazione

Questo documento elenca i problemi trovati nel notebook originale
(`Quadruplet_Network_for_attribute_recognition_and_person_re-ID.ipynb`, versione
`29cbb3c`), come sono stati corretti, e cosa cambia nella nuova versione
(`Quadruplet_Network_v2_improved.ipynb`).

La versione originale resta consultabile in git: `git show 29cbb3c:Quadruplet_Network_for_attribute_recognition_and_person_re-ID.ipynb`.

---

## 1. Bug che invalidano i risultati

Questi cinque non sono inefficienze: rendono privi di significato i numeri
riportati e i file di submission prodotti.

### 1.1 Immagini e label completamente disallineate — `MarketDataset.__getitem__`

```python
img_path = os.path.join(self.root_dir, self.list_dir[index])       # dal listing della cartella
anchor_image = Image.open(img_path)
attributes = torch.tensor([int(self.annotations.iat[index, i]) ...])  # dal dataframe
raw_ID = self.annotations.iat[index, 1]
```

`self.list_dir = sorted(os.listdir(root_dir))` è la lista di **tutti** i 12.989
file di `train/`, mentre `self.annotations` è un **sottoinsieme** di `complete_df`
(9.366 righe per il train, con `random_state=42`) costruito iterando
`os.listdir()` in ordine arbitrario.
Le due sequenze non hanno nessuna relazione: l'immagine all'indice *i* veniva
accoppiata agli attributi e all'identità di una persona diversa.

Conseguenze a cascata:

- la rete è stata addestrata su label permutate casualmente: la loss sugli
  attributi non era ottimizzabile oltre la frequenza della classe maggioritaria;
- la quadruplet loss campionava positivo e negativi in base a un `raw_ID` che non
  era l'ID dell'anchor, quindi anche le coppie "positive" erano casuali;
- `train_ds` e `valid_ds` hanno `root_dir = train_path` entrambe e iterano lo
  stesso `list_dir`, cioè **le stesse immagini**: la separazione train/validation
  per identità non aveva alcun effetto.

**Fix:** la lista dei file viene presa dal dataframe
(`self.annotations['image_name'].tolist()`), non dal filesystem. Allineamento
garantito per costruzione. Vedi anche `mode='folder'` per le cartelle non
etichettate.

### 1.2 Il modello salvato è inizializzato a caso

```python
if save_model:
    model = Backbone(attribute_map=att_map)     # rete NUOVA
    torch.save(model.state_dict(), 'model')     # pesi CASUALI
```

`main()` addestra `net` e poi salva i pesi di un'altra istanza appena creata. La
cella successiva ricarica quel file e usa `model` per generare
`classification_test.csv` e `reid_test.txt`:

```python
model = Backbone(attribute_map=att_map)
torch.save(model.state_dict(), 'model')     # sovrascrive di nuovo
model.load_state_dict(torch.load("model"))
```

**Entrambe le submission sono state generate da una rete non addestrata.**

**Fix:** si salva `net.state_dict()`, in un checkpoint che contiene anche
`att_map`, epoca, mAP e config; si tiene il *best* su mAP di validazione.

### 1.3 mAP misurata su feature mescolate

```python
validation_test_loader = DataLoader(validation_test_ds, batch_size,
                                    shuffle = True, drop_last=True)
```

La ground truth (`get_ground_truth`) è indicizzata per **posizione di riga** del
dataframe, mentre le feature della gallery escono dal loader in ordine
**mescolato** e con l'ultimo batch incompleto **scartato**. La matrice di
similarità confrontava quindi le feature di un'immagine con l'identità di
un'altra: la mAP stampata a ogni epoca era rumore.

Inoltre `validation_test_ds` usava `train_tfms`, cioè applicava random crop e
color jitter alle immagini di gallery durante la valutazione.

**Fix:** `shuffle=False, drop_last=False` su tutti i loader di valutazione
(con commento esplicito sul perché), `valid_tfms` in valutazione, e
`get_ground_truth` riscritta in termini di posizioni esplicite (e O(N+M) invece
di O(N·M)).

### 1.4 Dropout attivo durante `eval()`

```python
x = F.dropout(x, p=0.5)   # "this to keep into account for the net.eval/net.train"
```

`torch.nn.functional.dropout` ha `training=True` come default: non guarda lo stato
del modulo. Il commento dice l'opposto di quello che il codice fa. Ogni score di
validazione, ogni mAP e ogni predizione submittata è stata calcolata con metà del
bottleneck azzerata casualmente.

**Fix:** `nn.Dropout` come sottomodulo.

### 1.5 CSV di submission con le righe scambiate

```python
attr_dataframe = pd.DataFrame(data=attributes,
                              index=list(os.listdir(config['test_path'])), ...)
```

Le predizioni arrivano dal `test_loader`, che itera `sorted(os.listdir(...))`;
l'indice del CSV è invece un `os.listdir()` indipendente, il cui ordine è
arbitrario (dipende dall'ordine di creazione nella directory). Ogni riga del CSV
portava le predizioni di un'altra immagine.

**Fix:** l'indice è `test_ds.list_dir`, la stessa lista ordinata usata dal loader.

---

## 2. Bug di correttezza secondari

| # | Problema | Fix |
|---|---|---|
| 2.1 | `RandomCrop(32, padding=4)` dopo `Resize((224,224))`: la rete vede una toppa 32×32 di un pedone ingrandito, il 98% dell'immagine è buttato | `RandomCrop(image_size, padding=8)` — padding e crop *alla stessa dimensione* |
| 2.2 | `get_optimizer(net, lr, wd, momentum)` chiama `SGD(params, lr, momentum)`: il terzo argomento posizionale di SGD è `momentum`, quindi riceveva il weight decay (1e-6). Momentum reale = 1e-6, weight decay = 0 | argomenti passati per nome |
| 2.3 | Accuracy in `train()`: `predicted[i][1].eq(targets.transpose(0,1)[i]).sum()` confronta **la predizione della 2ª immagine del batch** con tutta la colonna dei target (broadcast) | contatore per-attributo in `AttributeMeter` |
| 2.4 | Il loop `accuracy_tot` è annidato *dentro* il loop per-attributo e riusa `i` come variabile, oscurando l'indice esterno | calcolo separato in `AttributeMeter` |
| 2.5 | `cumulative_loss/samples` mentre `loss.item()` è già una media per batch: la loss stampata è quella vera divisa per `batch_size` | somma pesata per batch size, divisa per il numero di campioni |
| 2.6 | `prob_pred_label` invoca ogni testa **3 volte in più** (`sigmoid(head(x)) if head(x).size()[1]==1 else softmax(head(x))`): 4 valutazioni × 29 teste per batch, e con dropout attivo le probabilità non corrispondono ai logit usati per la loss | logit calcolati una volta, probabilità derivate |
| 2.7 | `positive_list = [...][1:]` scarta un positivo arbitrario e può comunque estrarre l'anchor stesso (d=0 gratis); con `[1:]` un ID con 2 immagini restava con 1 candidato | l'anchor è escluso esplicitamente, nessun elemento scartato |
| 2.8 | I due negativi potevano coincidere (`random.sample` su una lista con duplicati logici) e `negative_list[1:]` scartava un candidato | campionamento con vincolo esplicito su indici distinti e ID ≠ anchor |
| 2.9 | `att_map` derivato da `complete_df.nunique()`: un attributo costante in uno split diventa silenziosamente binario, uno con una classe assente cambia dimensione della testa | cardinalità lette dallo schema delle annotazioni |
| 2.10 | `complete_df.columns[13:21]` / `[21:-1]` per i gruppi di colori: slice posizionali che funzionavano per coincidenza | selezione per nome (`startswith('up')` / `startswith('down')`) |
| 2.11 | `log_values` scrive `Total accuracy` senza `step`: ogni epoca sovrascrive lo stesso punto in TensorBoard | `step` passato |
| 2.12 | `Evaluator.evaluate_map` divide per `len(gt_set)`: `ZeroDivisionError` per una query senza immagini della stessa identità in gallery | le query senza ground truth vengono filtrate a monte, più guardia nell'evaluator |
| 2.13 | `open("reid_test.txt","a")`: rieseguendo la cella il file di submission accumula blocchi duplicati | `'w'` |
| 2.14 | `random.randint(0, n)` può restituire `n` → `IndexError` | `random.randrange(n)` |
| 2.15 | La griglia di visualizzazione è fissa a 15 colonne mentre `mAP_rank=20`: 5 immagini recuperate non venivano mai disegnate | `len(imgs)` colonne |
| 2.16 | `Image.open` senza `.convert('RGB')`: un jpg grayscale produce un tensore a 1 canale e fa crashare `Normalize` | `.convert('RGB')` |
| 2.17 | `optimizer.zero_grad()` chiamato *dopo* `step()`: funziona solo perché l'ordine è ciclico, ma con un `continue` o un'eccezione i gradienti di due batch si sommano | `zero_grad(set_to_none=True)` all'inizio dell'iterazione |
| 2.18 | `pretrained=True` rimosso nelle torchvision recenti | `weights='DEFAULT'` con fallback |
| 2.19 | `image_feature` senza `torch.no_grad()` proprio (dipendeva dal chiamante) e accumulava tutte le feature in VRAM | decoratore `@torch.no_grad()`, feature spostate su CPU |

---

## 3. Problemi di efficienza

| Problema | Impatto | Fix |
|---|---|---|
| `os.environ['CUDA_LAUNCH_BLOCKING'] = "1"` | serializza ogni lancio di kernel: throughput ~dimezzato senza motivo (è un flag di debug) | rimosso |
| 4 forward pass separati per batch (anchor, positivo, 2 negativi), ognuno con il suo grafo | ~4× overhead Python e di lancio kernel; i pesi sono condivisi, quindi concatenare dà gradienti identici | un solo forward su `cat([a,p,n1,n2])` |
| `positive_list`/`negative_list` ricostruite con una scansione completa del dataframe a **ogni** `__getitem__` | O(N) per campione × 4 immagini × 9.352 campioni per epoca: dominava il tempo di epoca | dizionario `ID → posizioni` precalcolato |
| `test()` ricalcola le feature di gallery+query **dentro** la funzione, e `main()` la chiama anche sul `train_loader` | più che raddoppia il costo di un'epoca per un numero scartato | mAP separata da `test()`, chiamata solo dove serve |
| Matrice di similarità costruita con un loop Python su 2.248 query × 19.679 gallery | ~45M `cosine_similarity` in Python | feature L2-normalizzate → un `matmul` a blocchi |
| Costruzione di `complete_df` con `csv.loc[csv['id']==ID]` per ognuno dei 12.989 file | O(N·M) | un `merge` |
| `image_size=(224,224)` su immagini sorgente 64×128 | aspect ratio distrutto **e** 1.5× i FLOP di 256×128 | `(256,128)`, standard per re-ID |
| fp32 | | AMP + `channels_last` |
| `num_workers=4` senza `pin_memory`, senza `persistent_workers` | i worker vengono ricreati a ogni epoca | `pin_memory=True`, `persistent_workers=True`, `num_workers` da `os.cpu_count()` |

---

## 4. Problemi metodologici

Non sono bug, ma limitano quello che si può concludere dai risultati.

1. **Accuracy media sugli attributi come metrica unica.** Gli attributi sono
   fortemente sbilanciati: `hat` è positivo sul 2.7% delle immagini, `uppurple`
   sul 3.3%, e `age` ha l'82% dei campioni in una sola delle 4 classi. Una testa
   che risponde sempre con la classe maggioritaria ottiene il 97% su `hat`. La
   "total accuracy" riportata è quindi dominata dalla classe maggioritaria; la v2
   riporta anche **balanced accuracy** e **macro-F1** per attributo.
2. **Le label degli attributi sono per identità, non per immagine.** I campioni
   indipendenti sono 751, non 12.989: le teste sugli attributi overfittano in
   fretta e lo split *deve* essere per identità (questo l'originale lo faceva
   correttamente, ed è la scelta giusta).
3. **Negativi casuali nella quadruplet loss.** Dopo poche epoche un negativo
   estratto a caso soddisfa già il margine e contribuisce gradiente nullo. La v2
   usa un **PK sampler** + **batch-hard mining**.
4. **Loss euclidea su feature non normalizzate, ranking con cosine similarity.**
   Le due geometrie non coincidono e i margini fissi (2.0 / 1.0) su distanze non
   limitate sono arbitrari. Ora le feature sono L2-normalizzate anche nella loss.
5. **Nessuna baseline.** Senza il numero di una ResNet18 pre-addestrata senza
   fine-tuning non si può dire quanto contribuisca l'addestramento.
6. **Nessun learning rate schedule, nessun early stopping, nessun seed.** Le run
   non erano riproducibili né confrontabili.
7. **Filtro per camera assente nella valutazione re-ID.** La convenzione di
   Market-1501 esclude dalla gallery le immagini della stessa identità **e** della
   stessa camera della query, altrimenti si premia il matching di sfondo. Qui si è
   mantenuto il comportamento dell'`Evaluator` fornito (che è la metrica valutata),
   ma va tenuto presente confrontando i numeri con la letteratura.

---

## 5. Cosa aggiunge la v2

Il notebook `Quadruplet_Network_v2_improved.ipynb` è una riscrittura, non una
patch. Impianto:

- **PK sampler** (`P` identità × `K` immagini) + **quadruplet loss con batch-hard
  mining**: positivo più lontano, negativo più vicino, e per il secondo termine la
  coppia di negativi di identità diverse più vicina. Un solo forward pass per step.
- **BNNeck** (Luo et al., *Bag of Tricks for Person Re-ID*): la loss metrica lavora
  sulla feature pre-BatchNorm, i classificatori su quella post-BatchNorm. Le due
  famiglie di loss vogliono geometrie diverse e sullo stesso vettore si
  ostacolano.
- **Testa di classificazione delle identità** (751 classi) con label smoothing:
  supervisione ID + metrica insieme è la ricetta standard, e da sola vale più di
  qualunque tuning della quadruplet.
- **GeM pooling** e **`last_stride=1`** (feature map 16×8 invece di 8×4).
- **Loss sugli attributi pesata** per frequenza di classe (`pos_weight` limitato a
  10× per non far esplodere gli attributi rari), pesi calcolati **solo sul train**.
- **AdamW** con lr 10× più basso sul trunk pre-addestrato, **warmup lineare +
  cosine decay** per step, grad clipping, AMP, `channels_last`, gradient
  accumulation opzionale.
- **Random erasing**, flip, pad+crop, jitter contenuto.
- **Metriche complete**: mAP@20 (la metrica valutata), mAP full, CMC rank-1/5/10,
  accuracy / balanced accuracy / macro-F1 per attributo.
- **Flip test** (media della feature dell'immagine e della sua speculare).
- **Checkpointing** best-su-mAP e **resume**.
- **Sanity check**: overfit di 8 identità in pochi minuti. È il controllo che
  avrebbe scoperto il bug 1.1 subito.
- **Cella di misura della VRAM** per scegliere il preset sulla propria GPU.

---

## 6. Note hardware — RTX 4070 Laptop 8 GB

Sì, gira, con margine. La rete è piccola (ResNet18 = 11M parametri, ResNet50 =
25M): il collo di bottiglia è la memoria delle attivazioni, che scala con il
numero di immagini per step.

| configurazione | immagini/step | risoluzione | AMP | VRAM di picco |
|---|---|---|---|---|
| v1 così com'è (ResNet18, 4 forward × 48, 224×224, fp32) | 192 | 224×224 | no | ~6.5 GB — **entra ma al limite** |
| v1 corretto (ResNet18, 1 forward × 128, 256×128, AMP) | 128 | 256×128 | sì | ~2.2 GB |
| v2 preset default (ResNet50, `last_stride=1`, 16×4) | 64 | 256×128 | sì | ~5.2 GB |
| v2 ResNet50 senza AMP | 64 | 256×128 | no | ~9.5 GB — **OOM** |
| v2 preset leggero (ResNet18, `last_stride=2`, 24×4) | 96 | 256×128 | sì | ~1.9 GB |

Le stime sono da calcolo analitico (attivazioni ≈ 28 MB/immagine per ResNet18 a
224×224 in fp32, scalate per risoluzione, precisione e profondità); il notebook v2
contiene la funzione `measure_peak_memory(cfg)` per misurarle davvero sulla tua
GPU prima di lanciare il training.

Su 8 GB **su laptop** conta anche che il desktop/browser occupa 0.5–1 GB: è il
motivo per cui la configurazione originale, che sulla carta entrerebbe, può
andare in OOM. Con AMP il margine c'è comunque.

**Tempi attesi** (RTX 4070 Laptop, che ha circa il 35% del budget di potenza di
una 4070 desktop): ~2–3 min/epoca per il preset default, quindi **1–1.5 h per 30
epoche**. Due avvertenze:

1. Il collo di bottiglia probabile è il **decoding JPEG**, non la GPU: 12.989 jpeg
   64×128 per epoca. Tieni `num_workers` a 6–8. Se `nvidia-smi dmon` mostra la GPU
   sotto l'80% di utilizzo, sei CPU-bound.
2. **Throttling termico**: una GPU laptop non sostiene il boost clock. Alimentatore
   collegato e profilo prestazioni, altrimenti aspettati 30–40% di immagini/s in
   meno dopo i primi minuti.

Se vai in OOM: abbassa prima `p_identities` (tieni `k_instances=4`, è quello che
rende possibile il mining), poi metti `last_stride=2`, poi scendi a `resnet34`.
`grad_accum=2` mantiene il batch efficace dimezzando la memoria.
