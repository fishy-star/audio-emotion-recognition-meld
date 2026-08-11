# SER_GAT Pipeline Walkthrough

Reference document for the audio-only Speech Emotion Recognition pipeline on
MELD. Reflects the code as it currently stands after the hygiene fixes,
oversampling reconciliation, and exploratory `gat_context_v2` work done in
this project. Written for understanding, not as a spec — if the code and this
document ever disagree, the code is right.

---

## 1. Data & Embeddings

**Raw source:** `data/raw/MELD.Raw/` — the original MELD dataset (train/dev/test
`*_sent_emo.csv` label files + per-split video archives). `utils/extract_audio.py`
pulls mono 16kHz `.wav` audio out of the MELD video splits into
`data/processed/audio/{train,dev,test}/dia<N>_utt<M>.wav`.

**Embedding extraction:** `utils/extract_embeddings.py` loads
`facebook/wav2vec2-base-960h` (`Wav2Vec2Model` + `Wav2Vec2Processor`), runs each
utterance's waveform through it, and mean-pools `last_hidden_state` over time to
get a single 768-dim vector per utterance. For each split it writes:
- `data/embeddings/{split}_embeddings.npy` — shape `(N, 768)`, one row per utterance
- `data/embeddings/{split}_metadata.csv` — `filename, dialogue_id, utterance_id, emotion` in the same row order as the `.npy` file (order/alignment is positional, not by an explicit join key — this is intentional and has been verified correct, but it's why nothing downstream ever re-sorts or filters one file without the other)

`fix_speaker_metadata.py` merges a `speaker` column into `train_metadata.csv` and
`dev_metadata.csv` from the raw MELD CSVs. It was originally only run for
train/dev — `test_metadata.csv` had no `speaker` column, which has since been
fixed by applying the identical merge (same raw MELD test CSV as source,
`(dialogue_id, utterance_id)` join key, `how="left"`) directly against
`test_metadata.csv`. Verified: row count unchanged (2,610), schema now matches
train/dev exactly, zero missing values, row order/identity untouched (only the
`speaker` column was added). The pre-fix file is preserved at
`archive/data_backups/test_metadata.csv.pre_speaker_fix`.

**However — re-running the test evaluation after this fix produced
byte-identical results** for every model, including the two that consume
`speaker` (`gat_context`, `gat_context_v2`). That's not because the fix didn't
take — it's because of a separate, deeper bug described in §2.4: the
same-speaker edge logic in `utils/graph_context.py` never actually adds a new
edge, on any split, whether `speaker` data is present or not. The
missing-column gap and this loop-range bug are two independent issues; fixing
the first exposed that the second had been masking it the whole time.

`utils/extract_embeddings_multilayer.py` is a separate, one-off script that
extracts all 13 wav2vec2 hidden-state layers per utterance (used only for the
wav2vec2-layer-fusion experiment tried while building `gat_context_v2`, which
did not outperform the DropEdge alternative and is not part of the production
pipeline — output lives at `data/embeddings/{split}_embeddings_layers.npy` if
present, but no live script depends on it).

**`scaler.pkl` and `label_encoder.pkl` — fit-once, transform-everywhere:**

Both artifacts are created by `train_mlp.py` (the first script in run order)
and reused everywhere else:

| Artifact | Created by | Fit on | Loaded by |
|---|---|---|---|
| `data/embeddings/scaler.pkl` (`sklearn.StandardScaler`) | `train_mlp.py`, via `scaler.fit_transform(X_train)` | train embeddings only | `train_bilstm.py`, `train_gat.py`, `train_gat_context.py`, `train_gat_context_v2.py` — all via `.transform()` only, never refit |
| `data/embeddings/label_encoder.pkl` (`sklearn.LabelEncoder`) | `train_mlp.py`, via `joblib.dump()` after `encoder.fit_transform(train_meta["emotion"])` | train labels only | the other four scripts, via `joblib.load()` + `.transform()` (in the GAT scripts, passed into `build_graphs(..., encoder=shared_encoder)` so `build_graphs` skips its own internal fit) |

Confirmed as implemented: `train_mlp.py` **must run first** — the other four
hard-depend on `data/embeddings/label_encoder.pkl` existing and will raise
`FileNotFoundError` otherwise. All five scripts print `.classes_` right after
obtaining the encoder; verified identical order across all five
(`anger, disgust, fear, joy, neutral, sadness, surprise`).

---

## 2. Per-model walkthrough

### 2.1 `train_mlp.py` — baseline (`models/mlp.py`)

- **Input:** utterance-level. Each row of the scaled 768-dim embedding is one
  independent training example — no dialogue/context structure at all.
- **Architecture:** `Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→7)`.
  The simplest model in the pipeline, no normalization layer on the input.
- **Loss:** `FocalLoss(alpha=weights, gamma=2.0)`, a hand-rolled focal loss
  class (defined identically at the top of all five scripts) — standard
  cross-entropy with a `(1-p)^gamma` down-weighting of easy examples, plus
  per-class `alpha` weights from `sklearn.compute_class_weight("balanced", ...)`
  on the train label distribution. Weights are computed, moved to `device`,
  and passed straight into `FocalLoss(alpha=weights, ...)` — confirmed reaching
  the loss (this was a real bug earlier in the project: weights were computed
  but silently never passed into `nn.CrossEntropyLoss()`; now fixed).
- **Optimizer / schedule:** `AdamW(lr=5e-4, weight_decay=1e-3)`. **No LR
  scheduler** — flat LR for the whole run. This is the only one of the five
  with no scheduler at all (BiLSTM also has none; both are pre-existing, not
  something Part 1 was asked to add).
- **Checkpoint selection:** macro F1 on dev (`if macro_f1 > best_f1`).
- **Early stopping:** patience=5, added in the recent hygiene pass — previously
  this script had no early stopping and always ran the full fixed `epochs=10`.
- **Oversampling:** none — doesn't apply; there's no dialogue-graph unit to
  oversample at the utterance level.
- **Quirks:** first script in the dependency chain (creates the shared
  scaler/encoder); the only script that fits rather than loads them.

### 2.2 `train_bilstm.py` — sequential context (`models/bilstm.py`)

- **Input:** dialogue-level sequences. A custom `DialogueDataset` groups
  utterances by `dialogue_id`, sorts by `utterance_id`, and returns each whole
  dialogue as a `(seq_len, 768)` tensor + label sequence. `collate_fn` pads
  variable-length dialogues to the batch max (`pad_sequence`, labels padded
  with `-100` so they're excluded from loss/metrics).
- **Architecture:** 2-layer bidirectional LSTM (`hidden_dim=256`, so
  `512`-dim output after concatenating directions) → `Dropout(0.3)` →
  `Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→7)`, applied per-timestep
  across the padded sequence.
- **Loss:** same `FocalLoss(alpha=weights, gamma=2.0)` pattern, plus
  `ignore_index=-100` so padded positions don't contribute.
- **Optimizer / schedule:** `AdamW(lr=5e-4, weight_decay=1e-3)`, gradient
  clipping (`max_norm=1.0`). No scheduler.
- **Checkpoint selection:** macro F1 on dev.
- **Early stopping:** patience=5 (added in the hygiene pass, same as MLP —
  previously ran the full fixed `epochs=30` unconditionally).
- **Oversampling:** none — same reasoning as MLP; this script batches whole
  dialogues via a standard shuffled `DataLoader`, not a per-graph sampler.
- **Quirks:** batch size 8 (dialogues), vs. MLP's batch size 32 (utterances) —
  reasonable given the very different unit of batching, not an oversight.

### 2.3 `train_gat.py` — graph, immediate-neighbor chain (`models/gat.py`,
`utils/graph.py`)

- **Input:** one PyG graph per dialogue. `utils/graph.py`'s `build_graphs`
  sorts each dialogue's utterances by `utterance_id` and connects only
  self-loops + immediate neighbors (`i↔i+1`) — the simplest possible edge
  structure, no speaker information, no wider window.
- **Architecture (`EmotionGAT`, shared by all three GAT scripts):** 2 stacked
  `GATConv` layers (`heads=4`, `hidden_dim=128`), each followed by ELU, a
  **per-layer residual** (`res1`/`res2`, separate `nn.Linear` projections sized
  to match each layer's output), then `LayerNorm`, then dropout (0.4).
  Classifier head: `Linear(128→128) → LayerNorm → ReLU → Dropout → Linear(128→7)`.

  This per-layer-norm-and-residual design is the fix for a measured
  over-smoothing problem: the original architecture had only a single
  end-of-stack residual and no normalization between the two `GATConv` layers,
  which was empirically shown (via adjacent-utterance cosine similarity,
  measured directly on dev-set node representations) to collapse node
  representations together — mean adjacent similarity rose from 0.173
  (raw input) to 0.836 after 2 unnormalized GAT layers. The current per-layer
  norm+residual design reduces that to ~0.55, with macro F1 improving from
  0.19→0.22 in the controlled before/after test. **This fix lives in
  `models/gat.py` and is therefore shared by all three GAT-family scripts** —
  it is not specific to `train_gat.py`.
- **Loss:** `FocalLoss(alpha=weights, gamma=2.0)`, same pattern as above.
- **Optimizer / schedule:** `AdamW(lr=5e-4, weight_decay=1e-5)` — note the
  weight_decay is 100x weaker than MLP/BiLSTM's `1e-3`; this is a deliberate,
  if not deeply re-validated, difference for the graph models.
  `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)` **stepped on
  `macro_f1`** (fixed in the recent hygiene pass — previously stepped on
  `weighted_f1`, which didn't match the checkpoint-selection metric).
- **Checkpoint selection:** macro F1 on dev.
- **Early stopping:** patience=5 (pre-existing in this script, unlike MLP/BiLSTM).
- **Oversampling:** `WeightedRandomSampler` over dialogue-graphs — any graph
  containing ≥1 `disgust`/`fear` utterance gets sample-weight **3.5**, else
  `1.0`, drawn with replacement, `num_samples=len(train_data)` per epoch. This
  exists because disgust/fear were found to be predicted **zero times** by the
  model on dev without oversampling (a structural collapse, not just poor
  recall) — the 3.5x multiplier was chosen after comparing it against 2.0x and
  no-oversampling on the combined disgust/fear-recovery vs.
  accuracy/weighted-F1 trade-off. (This script briefly diverged to 2.0x during
  that comparison and was reconciled back to 3.5x in the recent hygiene pass
  to match the other two GAT scripts.)
- **Quirks:** trains one dialogue-graph per optimizer step (no PyG batching) —
  noisier gradient signal than a true mini-batched loader would give; this is
  a known, accepted characteristic of all three GAT scripts, not something
  flagged as broken.

### 2.4 `train_gat_context.py` — graph, wider context window (`models/gat.py`,
`utils/graph_context.py`)

- **Input:** same graph-per-dialogue setup as `train_gat.py`, but edges come
  from `utils/graph_context.py` instead: each utterance connects to up to 2
  utterances before/after (`window=2`). This ±2 window — not speaker
  information (see below) — is the actual "context" that distinguishes this
  script from `train_gat.py`.
- **Architecture, loss, optimizer, checkpoint/early-stopping/oversampling
  config:** identical to `train_gat.py` in every respect — same `EmotionGAT`
  class, same `FocalLoss`, same `AdamW(5e-4, 1e-5)`, same macro-F1 scheduler
  and checkpoint, same patience=5, same 3.5x oversampling. The *only*
  intentional difference from `train_gat.py` is the edge-building function.
- **Confirmed bug — same-speaker edges are dead code:** `utils/graph_context.py`'s
  `build_edge_index` has a speaker branch intended to connect same-speaker
  utterances beyond the context window:
  ```python
  for j in range(max(0, i - window), min(num_nodes, i + window + 1)):
      edges.add((i, j)); edges.add((j, i))          # window edges (always run)
  ...
  for j in range(max(0, i - window), i):             # speaker edges
      if speakers[i] == speakers[j]:
          edges.add((i, j)); edges.add((j, i))
  ```
  The speaker loop's `j` range (`[i-window, i)`) is a **strict subset** of the
  window loop's range (`[i-window, i+window]`) — every pair it could add is
  already in the edge set. Discovered empirically: after fixing the missing
  `speaker` column on test (see §1), re-running `evaluate_test_set.py`
  produced byte-identical predictions/confusion matrices to the pre-fix run,
  for both `gat_context` and `gat_context_v2`. This means same-speaker edges
  have never contributed anything — not just on test (where `speaker` was
  missing), but on train/dev too, where the column existed the whole time. The
  "wider context + speaker-aware" design this script was meant to test has, in
  practice, only ever been the ±2 window with no speaker signal.
  **Not fixed** — that would be a change to a training-script dependency
  (`utils/graph_context.py`) and would imply the current `gat_context`/
  `gat_context_v2` checkpoints no longer reflect the intended architecture and
  should be retrained; left as a known, documented bug pending a separate
  decision on whether to fix + retrain.
- On the one completed dev-tuning run in this project, this script's macro F1
  (0.199, later 0.219 in a re-run) did **not** clearly and consistently beat
  the plain chain graph's — a genuine, reported result rather than an assumed
  win for "more context." In light of the bug above, that comparison was
  always effectively "±2 window vs. ±1 chain," never "window+speaker vs. chain."

### 2.5 `train_gat_context_v2.py` — context graph + DropEdge (`models/gat.py`,
`models/gat_v2.py`, `utils/graph_context.py`)

- **Input / architecture / loss / optimizer / checkpoint / oversampling:**
  identical to `train_gat_context.py` — same edge builder, same `EmotionGAT`,
  same `FocalLoss`, same `AdamW`, same macro-F1 scheduler/checkpoint, same
  3.5x oversampling. This also means it inherits §2.4's same-speaker-edges
  dead-code bug — this script's graphs are, in practice, the same ±2 window
  as `train_gat_context.py` with no speaker signal, regardless of the
  `speaker` column's presence.
- **The one deliberate addition: DropEdge, `p=0.15`.** During training only,
  `models.gat_v2.drop_edges(edge_index, p=0.15, training=True)` randomly drops
  ~15% of non-self-loop edges each optimizer step (self-loops are always kept)
  before the forward pass — a structural regularizer against overfitting to
  specific dialogue edge patterns. At eval time the full, undropped graph is
  used (`training=False` behavior — DropEdge never touches dev/test forward
  passes).
- **Provenance:** this is the one survivor of a broader exploration pass that
  also tried `GATv2Conv`, `PairNorm`, Jumping Knowledge, sinusoidal positional
  encoding, wav2vec2 13-layer softmax fusion (`extract_embeddings_multilayer.py`),
  and combinations of the above — all documented as negative or neutral
  results at the time (e.g. `GATv2Conv` alone: macro F1 0.187 vs. baseline
  0.199; DropEdge+positional-encoding combined: 0.197, worse than DropEdge
  alone's 0.215). `models/gat_v2.py` still contains the unused
  `FlexibleEmotionGAT` class from that exploration (GATv2/PairNorm/JK/pos-enc
  toggle support) — **it is not used by any of the five production scripts**;
  only its standalone `drop_edges()` function is imported and used, by this
  script alone.
- **Reproducibility:** this is the only one of the five scripts with an
  explicit seed (`SEED = int(os.environ.get("SEED", 42))`, `torch.manual_seed`
  + `np.random.seed`) — added because a single unseeded run of this exact
  config swung between macro F1 0.19 and 0.21 across repeats, which is why the
  seed exists here but not (yet) in the other four.
- **Test-set result:** did not show a clear advantage over `train_gat_context.py`
  or `train_gat.py` in the one official test-set pass (see §3) — reported as a
  lateral move, not chased further.

---

## 3. Evaluation pipeline

**`evaluate_test_set.py`** is the single, one-shot script that evaluates all
five trained checkpoints against the held-out MELD test split
(`data/embeddings/test_embeddings.npy` / `test_metadata.csv`, 2,610
utterances) — confirmed by `grep` to be untouched by any of the five training
scripts, `utils/graph*.py`, or `models/*.py` before this evaluation existed.

What it does, in order:
1. Loads the shared `scaler.pkl` (`.transform()` only) and `label_encoder.pkl`
   (`joblib.load`) — same artifacts every training script uses, so labels and
   feature scale are consistent with training.
2. Prints a warning if the `speaker` column is missing on test before
   evaluating the two `graph_context`-based models. `test_metadata.csv` now
   has this column (§1), so this warning no longer fires — but per §2.4, the
   column's presence doesn't actually change those two models' graphs either
   way, due to the separate same-speaker-edges bug.
3. Re-derives each model's own input structure at inference time: MLP gets
   flat utterance embeddings; BiLSTM gets padded per-dialogue sequences built
   the same way `DialogueDataset` does; the three GAT models get graphs built
   via each one's own edge-building function (`utils.graph.build_graphs` for
   `gat`, `utils.graph_context.build_graphs` for `gat_context` and
   `gat_context_v2`), with the shared label encoder passed in so nothing
   refits.
4. Loads each model's `checkpoints/{name}/{name}_best.pt` (or `gat_best.pt`
   inside the relevant subfolder) and runs a forward pass in eval mode — no
   DropEdge, no dropout active, no training-time randomness.
5. For each model, computes accuracy, macro F1, weighted F1, and a full
   7-class `precision/recall/F1/support` report + confusion matrix, saved to
   `outputs/test_evaluation/{model}_classification_report.txt` and
   `{model}_confusion_matrix.npy`.
6. Writes one consolidated table across all five models to
   `outputs/test_evaluation/consolidated_results.json` and
   `consolidated_results.txt`.

This script is meant to be run once per evaluation event, not repeatedly to
chase a better number. It has in fact been run twice: once as the original
official test-set evaluation, and once more after the `test_metadata.csv`
speaker-column fix (§1) — an explicit, disclosed correction, not a silent
rerun. The second run's numbers were byte-identical to the first for all five
models (confirmed via `diff`/`cmp`), which is itself the finding that surfaced
the §2.4 dead-code bug. Both runs' full outputs are preserved: the current
`outputs/test_evaluation/` reflects the post-fix run, and the pre-fix run is
archived at `archive/data_backups/test_evaluation_pre_speaker_fix/`. Any
future re-run (e.g. after fixing §2.4's bug and retraining) should be treated
as a new evaluation event, not a correction of either of these two.
