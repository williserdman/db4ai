# db4ai

A compact research workspace for graph-based learning experiments. The repo contains multiple subprojects and utilities used for training, analysis, and reproducibility of experiments.

Main components

- `argon/BiGCN`: legacy BiGCN rumor-detection code and related datasets.
- `sulfur`: training pipelines, Optuna hyperparameter search, and analysis utilities (under `sulfur/src`).
- `data`: processed and raw datasets used by experiments (relative to the repository root).

Goals of this README

- Be runnable from any machine (no user-specific absolute paths).
- Point to per-subproject dependency instructions.
- Show common, repeatable commands using relative paths and virtual environments.

Prerequisites

- Python 3.8+ (3.9/3.10/3.11 recommended).
- `pip` or `poetry` to manage Python dependencies.
- Optional: CUDA + a compatible PyTorch installation if you want GPU training.

Installation (recommended)

1. Create and activate a virtual environment (from the repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

2. Install dependencies per subproject:

````bash
  pip install -r requirements.txt```
````

Data layout

- The `data/` directory is used to store raw and processed datasets. Many scripts expect datasets at `data/<dataset-name>/...`.
- If you keep datasets elsewhere, either create symlinks from `data/` to your dataset folder or set environment variables/CLI flags in the scripts to point to your dataset root.

Running common commands

- Use the repository root as your working directory and avoid absolute paths. Examples below assume the virtual environment is active.

- Example: run the analysis CLI (feature repair) using `sulfur/src` as a module path:

```bash
# from repo root
PYTHONPATH=sulfur/src python -m analysis.cli \
    --dataset ENGB \
    --checkpoint path/to/checkpoint.pt \
    --mode feature-repair \
    --plots-dir output/plots
```

- Example: run a main training script located at `sulfur/src/main.py` (if available):

```bash
python -m src.main --help
```

Notes and tips

- Prefer relative paths and project-local virtual environments; do not hard-code `/Users/...` paths.
- Use `--help` on CLI entry points to learn available flags (e.g., `--plots-dir`, `--max-epochs`, `--lr`).
- Many commands accept `--plots-dir` to save diagnostic plots and `--checkpoint` to load models.

Project structure (quick reference)

- `argon/` — legacy BiGCN code and helpers.
- `data/` — datasets and processed files (shared by experiments).
- `sulfur/` — main training/analysis code under `sulfur/src` and Optuna logs under `sulfur/optuna_logs`.

Contributing

- If you add datasets, place them under `data/` or update loaders to accept a configurable data root.
- Please open issues or PRs for reproducibility improvements, missing dependency notes, or clearer run scripts.

License

- No license is specified in this repository. Add a LICENSE file if you intend to publish or share the code publicly.

If you want, I can also:

- Add step-by-step run examples for a specific subproject (e.g., full training command for `sulfur`).
- Create a minimal `Makefile` or `scripts/` helpers to standardize common commands.

---

Updated to remove absolute, machine-specific paths and to use relative commands and virtual environments.
