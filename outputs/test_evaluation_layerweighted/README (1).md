# Audio-Only Emotion Recognition on MELD

Does conversational context help emotion recognition when using audio alone — no text, no video? This project builds and compares five models on the MELD dataset to find out, using wav2vec2 speech embeddings.

Most graph-based emotion recognition research is multimodal — it fuses text, audio, and video together, and text usually ends up carrying most of the result. That makes it hard to tell whether graph-based context actually helps with audio specifically, or whether it's just riding along on the text signal. This project isolates audio on its own to answer that more directly.

## Models

- `mlp` — no context, each utterance classified on its own
- `bilstm` — sequential context, reads the whole dialogue
- `gat` — graph attention, connects each utterance to its nearest neighbors
- `gat_context` — wider graph context, connects same-speaker utterances across the dialogue
- `gat_context_v2` — same as above, with DropEdge regularization

## Results

Test-set macro F1:

| Model | Baseline | With multi-layer wav2vec2 | Δ |
|---|---|---|---|
| mlp | 0.2008 | 0.2763 | +0.0755 |
| bilstm | 0.1929 | 0.2649 | +0.0720 |
| gat | 0.2013 | 0.2762 | +0.0749 |
| gat_context | 0.1911 | 0.2686 | +0.0775 |
| gat_context_v2 | 0.1797 | 0.2623 | +0.0826 |

Adding conversational context doesn't clearly improve results — the simplest models score about the same as the most complex ones. This matches what other researchers have found in text-based emotion recognition on MELD, extended here to audio only.

Switching from single-layer to multi-layer wav2vec2 pooling (weighting all 13 layers instead of just the last one) gave a consistent improvement across every model, around 7-8 points of macro F1.

## Does hand-crafted acoustic signal move the ceiling?

The above results all come from wav2vec2, a learned representation. A separate
question: is the ~0.26-0.28 macro-F1 ceiling a *representation* limit (wav2vec2
missing paralinguistic signal), or a *data/task* limit (MELD's background
score/laugh-track noise, class imbalance, and short utterances capping any
acoustic representation)? To test this, a hand-crafted acoustic feature set was
built independently of wav2vec2 — GFCC, MFCC, RMS energy, `pyin` F0, chroma,
spectral flux/bandwidth/centroid, ZCR, and delta/delta-delta of the cepstral
features (96 frame-level features; mean+std flat-aggregate = 192-dim, or raw
frame sequences for the BiLSTM tier) — and run through two tiers:

- **Acoustic-only**: wav2vec2 replaced entirely by the acoustic feature set.
- **Fusion**: the acoustic flat-aggregate concatenated onto the layer-weighted
  wav2vec2 vector (768+192 = 960-dim). The BiLSTM tier fuses the pooled
  acoustic summary onto each utterance's wav2vec2 vector rather than at the
  raw frame level, since the dialogue-level BiLSTM already expects one vector
  per utterance and this needed no new architecture beyond input width.

Test-set macro F1:

| Model | Baseline (wav2vec2) | Acoustic-only | Δ vs baseline | Layer-weighted (wav2vec2) | Fusion | Δ vs layer-weighted |
|---|---|---|---|---|---|---|
| mlp | 0.2008 | 0.2123 | +0.0115 | 0.2763 | 0.2767 | +0.0004 |
| bilstm | 0.1929 | 0.1617 | -0.0312 | 0.2649 | 0.2574 | -0.0075 |
| gat | 0.2013 | 0.2036 | +0.0023 | 0.2762 | 0.2615 | -0.0147 |
| gat_context | 0.1911 | 0.1873 | -0.0038 | 0.2686 | 0.2664 | -0.0022 |
| gat_context_v2 | 0.1797 | 0.1960 | +0.0163 | 0.2623 | 0.2459 | -0.0164 |

**Acoustic-only lands roughly on par with the wav2vec2 baseline** — up for 3 of
5 architectures, down for 2, all within ~0.03 macro F1. A 192-dim hand-crafted
feature set gets to essentially the same place as a 768-dim pretrained
self-supervised representation, for this task.

**Fusion does not move the ceiling.** On dev, every fusion model scored
0.27-0.30 macro F1, apparently clearing the layer-weighted ceiling across the
board — but that gain evaporated on the held-out test set: 4 of 5 architectures
score *below* their layer-weighted-only counterpart, and mlp is flat (+0.0004).
The dev-set "improvement" looks like optimistic variance from tuning on a small
validation split rather than a genuine information gain, which only the
test set (never touched during training or checkpoint selection) can catch.

Read together with the acoustic-only result, this is a positive finding for
the project's central claim: concatenating a second, independently-noisy
acoustic representation on top of wav2vec2 doesn't unlock additional signal.
That's consistent with the ceiling being a genuine information limit in
MELD's audio channel — background score/laugh-track contamination, severe
class imbalance (~47% neutral), and short utterances — rather than a fixable
gap in wav2vec2's representation.

**The context-modeling gap does not reopen.** Under both new tiers, `mlp` (no
context at all) is tied for the best or is the single best architecture, and
the most elaborate context model (`gat_context_v2`) is the worst performer in
both tiers. Richer input didn't change which architecture wins — conversational
context still doesn't help audio-only ERC on MELD, regardless of whether the
input is wav2vec2, hand-crafted acoustic features, or both concatenated
together.

**GFCC does not dominate on this noisy TV audio.** A class-balanced feature
importance diagnostic (Information Gain, Random Forest, XGBoost, all with
balanced class weighting so the ~47% neutral class doesn't swamp the ranking)
was run on the flat acoustic features against the true labels. In prior
clean-corpus SER experiments (RAVDESS/EMOVO/SUBESCO/EMODB), GFCC's
noise-robust Gammatone filterbank tends to dominate importance rankings. Here
it doesn't: MFCC ranks higher (mean rank 92.1) than GFCC (mean rank 106.7)
across all three scorers. GFCC's noise-robustness advantage, at least as
measured by these three importance scorers, doesn't clearly transfer from
clean acted-emotion corpora to MELD's genuinely noisy TV audio. Full rankings:
`outputs/acoustic_feature_importance/feature_importance.csv`.

## Setup

```
pip install -r requirements.txt

python utils/extract_embeddings.py
python utils/extract_embeddings_multilayer.py

python training/train_mlp.py
python training/train_bilstm.py
python training/train_gat.py
python training/train_gat_context.py
python training/train_gat_context_v2.py
python evaluation/evaluate_test_set.py

python training/train_mlp_v2.py
python training/train_bilstm_v2.py
python training/train_gat_v2.py
python training/train_gat_context_lw.py
python training/train_gat_context_v2_lw.py
python evaluation/evaluate_test_set_layerweighted.py

python utils/extract_acoustic_features.py
python evaluation/acoustic_feature_importance.py

python training/train_mlp_acoustic.py
python training/train_bilstm_acoustic.py
python training/train_gat_acoustic.py
python training/train_gat_context_acoustic.py
python training/train_gat_context_v2_acoustic.py
python evaluation/evaluate_test_set_acoustic.py

python training/train_mlp_fusion.py
python training/train_bilstm_fusion.py
python training/train_gat_fusion.py
python training/train_gat_context_fusion.py
python training/train_gat_context_v2_fusion.py
python evaluation/evaluate_test_set_fusion.py
```

## Structure

```
training/       training scripts
evaluation/     evaluation scripts
models/         model architectures
utils/          embedding extraction, graph construction
checkpoints/    trained model weights
outputs/        results, classification reports, confusion matrices
archive/        earlier experiments not used in final results
```

## Dataset

[MELD](https://affective-meld.github.io/) (Poria et al., 2019)

## References

- Ghosal et al. (2019). DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. EMNLP-IJCNLP.
- Hu et al. (2021). MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation. ACL-IJCNLP.
- Li et al. (2023). GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection. IEEE Trans. on Affective Computing.
- Pepino et al. (2021). Emotion Recognition from Speech Using wav2vec 2.0 Embeddings. Interspeech.
- Poria et al. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL.
