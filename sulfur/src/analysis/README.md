# Analysis Utilities

This folder contains CLI helpers for feature repair, reconstruction scoring, and the argon text-embedding workflow.

## Argon text repair (embed + de-embed)

Run the argon text workflow from the CLI:

```bash
PYTHONPATH=sulfur/src /usr/bin/env python -m analysis.cli \
  --mode argon-text-repair \
  --argon-edge-path argon/BiGCN/data/Twitter15/data.TD_RvNN.vol_5000.txt \
  --argon-text-path argon/source_tweets_t15.tsv \
  --text-embedding-model bow \
  --text-vocab-path sulfur/optuna_logs/argon_bow_vocab.json \
  --recon-method gae \
  --recon-top-k 50 \
  --export-csv sulfur/optuna_logs/argon_candidates.csv \
  --plots-dir sulfur/optuna_logs/analysis_plots
```

The exported CSV includes candidate nodes with low reconstruction confidence plus a nearest-neighbor text lookup
(used as the "de-embed" suggestion). Edit the `text` field and re-run with `--import-csv`:

```bash
PYTHONPATH=sulfur/src /usr/bin/env python -m analysis.cli \
  --mode argon-text-repair \
  --argon-edge-path argon/BiGCN/data/Twitter15/data.TD_RvNN.vol_5000.txt \
  --argon-text-path argon/source_tweets_t15.tsv \
  --text-embedding-model bow \
  --text-vocab-path sulfur/optuna_logs/argon_bow_vocab.json \
  --recon-method gae \
  --recon-top-k 50 \
  --export-csv sulfur/optuna_logs/argon_candidates.csv \
  --import-csv sulfur/optuna_logs/argon_candidates.csv \
  --plots-dir sulfur/optuna_logs/analysis_plots
```

Outputs:
- `text_embedding_pca_before_after.png` shows corrected nodes in embedding space.
- `feature_pca_before_after.png` is used by the feature-repair modes.

Notes:
- `--recon-method ae` uses autoencoder reconstruction error (higher is worse).
- `--recon-method gae` uses GAE confidence (lower is worse).
- `--use-faiss` accelerates de-embed lookups if FAISS is installed.
