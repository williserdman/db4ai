# db4ai

A compact research workspace for graph-based learning experiments with multiple subprojects, datasets, and analysis tools.

Repository map (full scan)

- `argon/`: dataset exploration and coverage tooling for BiGCN and Twitter15/16 text alignment.
  - `argon/BiGCN/`: upstream BiGCN code and data processing scripts.
  - `argon/check_message_coverage.py`: compares BiGCN cascade roots vs Kaggle tweet text IDs.
  - `argon/coverage_report.txt`: coverage summary output.
  - `argon/source_tweets_t15.tsv`: Kaggle text source for Twitter15.
- `sulfur/`: main training and analysis code.
  - `sulfur/src/`: training, models, loaders, and analysis utilities.
  - `sulfur/src/analysis/`: CLI utilities for feature repair and reconstruction workflows.
  - `sulfur/src/notebooks/`: exploration notebooks.
  - `sulfur/optuna_logs/`: Optuna trial outputs and plots.
- `data/`: datasets and processed splits used by experiments.

Key entry points

- Training orchestration: `sulfur/src/main.py`
- Optuna pipeline: `sulfur/src/optuna_trainer.py`
- Analysis CLI: `python -m analysis.cli` (see `sulfur/src/analysis/README.md`)
- BiGCN baseline runs: `argon/BiGCN/main.sh`

Prerequisites

- Python 3.8+ (3.9/3.10/3.11 recommended).
- `uv` for Python dependency management.
- Optional: CUDA + compatible PyTorch build if you want GPU training.

Installation (recommended, using `uv`)

1. Create and activate a virtual environment (from the repository root):

```bash
uv venv .venv
source .venv/bin/activate   # macOS / Linux
```

2. Install dependencies per subproject:

- Root environment snapshot (if you want the same environment used locally):

```bash
uv pip install -r requirements.txt
```

- BiGCN baseline (if you plan to run `argon/BiGCN` experiments):

```bash
uv pip install -r argon/BiGCN/requirements.txt
```

- `sulfur` (PEP 517/660 install from `pyproject.toml`):

```bash
uv pip install -e sulfur
```

Data layout

- The `data/` directory stores raw and processed datasets. Many scripts expect datasets at `data/<dataset-name>/...`.
- If you keep datasets elsewhere, create symlinks under `data/` or update loaders/flags to point at your data root.

Running common commands

- Use the repository root as your working directory and avoid absolute paths.

- Example: analysis CLI (feature repair) using `sulfur/src` as the module path:

```bash
# from repo root
PYTHONPATH=sulfur/src python -m analysis.cli \
  --dataset ENGB \
  --checkpoint path/to/checkpoint.pt \
  --mode feature-repair \
  --plots-dir output/plots
```

- Example: run the main training script:

```bash
python -m src.main --help
```

- Example: BiGCN baseline run (see `argon/BiGCN/readme.md` for details):

```bash
cd argon/BiGCN
sh main.sh
```

Results and figures

- Coverage analysis (BiGCN cascade roots vs Kaggle tweets):
  - Summary in `argon/coverage_report.txt`.
  - Latest coverage: 48.10% (1490/3098 cascades matched).

- Analysis plots (from CLI/notebook runs) in `sulfur/optuna_logs/analysis_plots/`:
  - `feature_drift_hist.png` (feature drift histogram)
  - `feature_pca_before_after.png` (PCA of target node features before/after repair)
  - `neighborhood_0_1hop.png` (local neighborhood correctness before/after repair)

You can view figures directly:

![Feature drift histogram](sulfur/optuna_logs/analysis_plots/feature_drift_hist.png)
![Feature PCA before/after](sulfur/optuna_logs/analysis_plots/feature_pca_before_after.png)
![Neighborhood before/after](sulfur/optuna_logs/analysis_plots/neighborhood_0_1hop.png)

Notebooks

- Exploration notebooks live in `sulfur/src/notebooks/`:
  - `graph_embed_repair_overfit.ipynb`
  - `graph_reconstruct_straight.ipynb`
  - `graph_reconstruct_w_gae.ipynb`

Notes and tips

- Prefer relative paths and project-local virtual environments; do not hard-code `/Users/...` paths.
- Use `--help` on CLI entry points to discover flags like `--plots-dir`, `--max-epochs`, and `--lr`.
- Many commands accept `--plots-dir` to save diagnostic plots and `--checkpoint` to load models.

Contributing

- If you add datasets, place them under `data/` or update loaders to accept a configurable data root.
- Please open issues or PRs for reproducibility improvements, missing dependency notes, or clearer run scripts.

License

- No license is specified in this repository. Add a LICENSE file if you intend to publish or share the code publicly.
