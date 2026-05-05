# db4ai workspace

This workspace contains two major codebases:

- [argon/BiGCN](argon/BiGCN): Graph rumor detection experiments and scripts.
- [sulfur](sulfur): Graph learning experiments, Optuna training, and analysis notebooks/tools.

## Quick map

- [argon/BiGCN](argon/BiGCN): Original BiGCN code and datasets.
- [sulfur/src](sulfur/src): Training, models, loaders, and analysis utilities.
- [data](data): Cached datasets and splits used by multiple experiments.

## sulfur: training and experiments

Entry points and configuration:

- Training orchestration: [sulfur/src/main.py](sulfur/src/main.py)
- CLI/config helpers: [sulfur/src/args.py](sulfur/src/args.py)
- Hyperparameter search: [sulfur/src/optuna_trainer.py](sulfur/src/optuna_trainer.py)

Typical run flow:

1. Select datasets and hyperparameters in [sulfur/src/main.py](sulfur/src/main.py).
2. Run the training script (see [sulfur/pyproject.toml](sulfur/pyproject.toml) for dependencies).
3. Analyze results and compare runs with logs in [sulfur/optuna_logs](sulfur/optuna_logs).

## Extracted notebook tools

Notebook logic was consolidated into reusable utilities under [sulfur/src/analysis](sulfur/src/analysis). Add [sulfur/src](sulfur/src) to your `PYTHONPATH` or `sys.path` to import them:

- Feature repair (node embedding delta optimization): [sulfur/src/analysis/feature_repair.py](sulfur/src/analysis/feature_repair.py)
- GAE and AE reconstruction helpers: [sulfur/src/analysis/reconstruction.py](sulfur/src/analysis/reconstruction.py)
- Counterfactual edge tests: [sulfur/src/analysis/counterfactual.py](sulfur/src/analysis/counterfactual.py)
- Shared accuracy helpers: [sulfur/src/analysis/metrics.py](sulfur/src/analysis/metrics.py)

### CLI quick run

Use the lightweight CLI to run the extracted tools without opening notebooks:

```bash
PYTHONPATH=sulfur/src /Users/williserdman/Documents/school/Spring2026/db4ai/db4ai/.venv/bin/python -m analysis.cli \
    --dataset ENGB \
    --checkpoint sulfur/optuna_logs/best_params/version_2/checkpoints/epoch=529-step=530.ckpt \
    --mode feature-repair \
    --plots-dir sulfur/optuna_logs/analysis_plots \
    --trust-checkpoint
```

Other modes:

- `feature-repair-all`: optimize a delta using all node labels (masked to the target set).
- `gae-confidence`: train a GAE and report per-node reconstruction confidence stats.
- `gae-khop-loss`: compute k-hop reconstruction loss around incorrect nodes.

### CLI reference

Basic usage:

```bash
PYTHONPATH=sulfur/src /Users/williserdman/Documents/school/Spring2026/db4ai/db4ai/.venv/bin/python \
    -m analysis.cli --dataset DATASET --checkpoint PATH --mode feature-repair
```

Core flags:

- `--dataset`: dataset key passed to the loader (e.g., `ENGB`, `Cora`).
- `--checkpoint`: path to a `MyModel` checkpoint.
- `--mode`: `feature-repair`, `feature-repair-all`, `gae-confidence`, `gae-khop-loss`.
- `--trust-checkpoint`: allow loading checkpoints that contain non-weight objects.

If you see `TypeError: cannot unpack non-iterable NoneType object`, re-run with:

- `--max-epochs 1 --log-every 1` to keep the repair run minimal
- `--plots-dir` to ensure plot outputs are enabled

Feature repair tuning:

- `--max-epochs`, `--lr`, `--weight-decay`, `--log-every`
- `--max-target-nodes`: cap number of incorrect nodes to repair.
- `--l2-to-original`: regularize repaired features toward original values.
- `--no-entropy-weighting`: disable entropy weighting for target loss.

Plots:

- `--plots-dir`: enable plot output (neighborhood before/after + feature drift).
- `--target-node`: node id to visualize (defaults to first fixed/target node).
- `--neighborhood-hops`: k-hop neighborhood depth for the plot.

When `--plots-dir` is provided, the CLI will emit:

- `neighborhood_<node>_<k>hop.png`: before/after correctness in the k-hop subgraph.
- `feature_drift_hist.png`: L2 drift histogram for target nodes.
- `feature_pca_before_after.png`: PCA scatter of target features before/after repair.

### Example: feature repair

```python
from analysis import FeatureRepairConfig, repair_features_overfit

config = FeatureRepairConfig(max_epochs=500, lr=0.1, use_entropy_weighting=True)
result = repair_features_overfit(model, graph_data, config)

print(result.metrics_base)
print(result.best_state["metrics"])
```

### Example: GAE reconstruction scoring

```python
from analysis import GAEConfig, train_gae, compute_node_recon_confidence

gae = train_gae(graph_data.x, graph_data.edge_index, GAEConfig())
node_conf = compute_node_recon_confidence(gae, graph_data.x, graph_data.edge_index)
```

### Example: counterfactual edge witnesses

```python
from analysis import find_counterfactual_witnesses

witnesses = find_counterfactual_witnesses(
    model=model,
    graph_data=graph_data,
    wrong_nodes=wrong_nodes,
    ground_truth=graph_data.y,
    predictions=predictions,
    probs=probs,
    ae_node_err=ae_node_err,
    graph_nx=G,
    top_neighbors_to_test=5,
)
```

## data layout

The [data](data) directory stores cached datasets and splits. Many scripts expect these datasets to exist locally. If you add new datasets, follow the same folder layout.

## Notebooks

Original exploration notebooks remain in [sulfur/src/notebooks](sulfur/src/notebooks) for reference. The core logic is now in [sulfur/src/analysis](sulfur/src/analysis).

## Future Work

This is extremely dependent on finding a dataset which fits the following limitations:

- needs raw features data (messages in plain text format for interpretation)
- graph structure

This has been rather challenging to find. Below I outline three potential paths forward:

1. scrape ourselves, one potential option is to scrape the data ourselves. it seems that generally there are not a set of datasets that fit our needs. scraping them ourselves could be a good avenue to publish a dataset and work with it for this project
2. knowledge graphs. the structure of these types of graphs is different than interaction graphs, however, there may be an avenue to explore here with graphs that are well labelled and have explicit plain text features
3. a promising avenue forward seems to be the combination of some twitter datasets. seen in the `argon` folder above the BiGCN paper created a graph datastructure of tweets. the original Twitter15/16 dataset from Kaggle can be combined with this one to get rich node labels. two key limitaitons here: firstly, there is not full coverage of the dataset, BiGCN nodes don't match the Kaggle nodes fully. secondly, the dataset is set up in many isolated tree like structures (like root post, then replies) rather than one cohesive graph.
