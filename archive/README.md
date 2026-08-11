# archive/

Historical, superseded work — not part of the current pipeline. Kept for
provenance, not maintained or re-run.

- **`train_proposed*.py`, `models/proposed*.py`, `utils/graph_typed*.py`** —
  earlier relation-aware graph experiments (typed edges, RGATConv, RGAT +
  speaker features) that preceded the `gat` / `gat_context` / `gat_context_v2`
  line described in `PIPELINE_WALKTHROUGH.md`. Superseded, not benchmarked
  against the current models under the same conditions.
- **`checkpoints/`, `outputs/`** — checkpoints and results for the above,
  matched by directory name (`proposed`, `proposed_v2`, `proposed_v3`,
  `proposed_rgat`, `proposed_rgat_speaker`, `proposed_typed`,
  `proposed_typed_v2`).
- **`pre_speaker_fix_checkpoints/`** — `gat_context` / `gat_context_v2`
  checkpoints from before the `test_metadata.csv` speaker-column fix
  (see `PIPELINE_WALKTHROUGH.md` §1). Superseded by the current
  `checkpoints/gat_context/` and `checkpoints/gat_context_v2/`.
- **`data_backups/test_evaluation_pre_speaker_fix/`** — the test-set
  evaluation run from before that same fix, kept alongside the current
  `outputs/test_evaluation/` for comparison. The two runs are byte-identical
  for `gat_context`/`gat_context_v2` — expected, and explained in
  `PIPELINE_WALKTHROUGH.md` §2.4.
