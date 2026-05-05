"""Analysis tools extracted from notebooks."""

from importlib import import_module

__all__ = [
    "eval_preds",
    "split_acc",
    "FeatureRepairConfig",
    "FeatureRepairResult",
    "repair_features_overfit",
    "repair_features_all_nodes",
    "summarize_drift",
    "AutoencoderConfig",
    "GAEConfig",
    "StandardAutoencoder",
    "GCNEncoder",
    "train_autoencoder",
    "train_gae",
    "compute_node_recon_confidence",
    "reconstruct_edges_by_threshold",
    "gae_k_hop_recon_loss",
    "remove_edge_both_directions",
    "find_counterfactual_witnesses",
]

_EXPORTS = {
    "metrics": {"eval_preds", "split_acc"},
    "feature_repair": {
        "FeatureRepairConfig",
        "FeatureRepairResult",
        "repair_features_overfit",
        "repair_features_all_nodes",
        "summarize_drift",
    },
    "reconstruction": {
        "AutoencoderConfig",
        "GAEConfig",
        "StandardAutoencoder",
        "GCNEncoder",
        "train_autoencoder",
        "train_gae",
        "compute_node_recon_confidence",
        "reconstruct_edges_by_threshold",
        "gae_k_hop_recon_loss",
    },
    "counterfactual": {"remove_edge_both_directions", "find_counterfactual_witnesses"},
}


def __getattr__(name: str):
    for module, names in _EXPORTS.items():
        if name in names:
            return getattr(import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
