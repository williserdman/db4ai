from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

_SULFUR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SULFUR_ROOT))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'matplotlib'. Activate the project's Python environment or install "
        "requirements before running this CLI."
    ) from exc

try:
    import networkx as nx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'networkx'. Activate the project's Python environment or install "
        "requirements before running this CLI."
    ) from exc

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Activate the project's Python environment or install "
        "requirements before running this CLI."
    ) from exc
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from loading.LightningGraphLoader import load_datasets
from loading.DatasetInfo import DatasetInfo
from models.MyModel import MyModel

from .feature_repair import FeatureRepairConfig, repair_features_all_nodes, repair_features_overfit
from .metrics import split_acc
from .reconstruction import GAEConfig, compute_node_recon_confidence, gae_k_hop_recon_loss, train_gae


def _load_graph(ds_name: str) -> Data:
    data = load_datasets([ds_name])[ds_name]
    dl = data.data.train_dataloader()
    return dl.dataset[0]


def _load_model(checkpoint_path: str, trust_checkpoint: bool) -> MyModel:
    if trust_checkpoint:
        torch.serialization.add_safe_globals([DatasetInfo])
        model = MyModel.load_from_checkpoint(checkpoint_path, weights_only=False)
    else:
        model = MyModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def _build_target_mask(
    model: torch.nn.Module,
    graph_data: Data,
    max_target_nodes: Optional[int],
) -> torch.Tensor:
    device = next(model.parameters()).device
    graph_data = graph_data.to(device)
    with torch.no_grad():
        logits, _ = model(graph_data)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

    _, correct_base = split_acc(
        pred,
        graph_data.y,
        graph_data.train_mask,
        graph_data.val_mask,
        graph_data.test_mask,
    )
    incorrect_mask = ~correct_base

    if max_target_nodes is None:
        return incorrect_mask

    eps = 1e-12
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
    target_indices = torch.where(incorrect_mask)[0]
    if target_indices.numel() == 0:
        raise RuntimeError("No incorrect nodes available for target selection.")

    order = torch.argsort(entropy[target_indices], descending=True)
    target_indices = target_indices[order[:max_target_nodes]]
    target_mask = torch.zeros_like(incorrect_mask, dtype=torch.bool)
    target_mask[target_indices] = True
    return target_mask


def _ensure_dir(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _select_plot_node(
    target_mask: torch.Tensor,
    correct_base: torch.Tensor,
    correct_repaired: torch.Tensor,
    target_node: Optional[int],
) -> int:
    if target_node is not None:
        return int(target_node)

    fixed = (~correct_base) & correct_repaired
    fixed_indices = torch.where(fixed)[0]
    if fixed_indices.numel() > 0:
        return int(fixed_indices[0].item())

    target_indices = torch.where(target_mask)[0]
    if target_indices.numel() == 0:
        raise RuntimeError("No target nodes available for plotting.")
    return int(target_indices[0].item())


def _plot_neighborhood(
    graph_data: Data,
    pred_base: torch.Tensor,
    pred_repaired: torch.Tensor,
    node_id: int,
    hops: int,
    out_dir: Path,
) -> None:
    subset, edge_index_k, _, _ = k_hop_subgraph(
        torch.tensor([node_id]),
        num_hops=hops,
        edge_index=graph_data.edge_index,
        relabel_nodes=True,
    )
    G = to_networkx(Data(edge_index=edge_index_k, num_nodes=subset.numel()), to_undirected=True)

    base_correct = pred_base[subset].eq(graph_data.y[subset])
    repaired_correct = pred_repaired[subset].eq(graph_data.y[subset])

    pos = nx.spring_layout(G, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, mask, title in (
        (axes[0], base_correct, "Before repair"),
        (axes[1], repaired_correct, "After repair"),
    ):
        colors = ["green" if mask[i].item() else "red" for i in range(len(mask))]
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=60, ax=ax, alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25)
        ax.set_title(f"{hops}-hop neighborhood: {title}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"neighborhood_{node_id}_{hops}hop.png", dpi=200)
    plt.close(fig)


def _plot_feature_drift(
    x_original: torch.Tensor,
    x_repaired: torch.Tensor,
    target_mask: torch.Tensor,
    out_dir: Path,
) -> None:
    drift = (x_repaired - x_original).norm(dim=1).detach().cpu()
    target_mask_cpu = target_mask.detach().cpu()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(drift[target_mask_cpu].numpy(), bins=40, alpha=0.75, color="tab:blue")
    ax.set_title("Feature repair drift (target nodes)")
    ax.set_xlabel("L2 drift")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_drift_hist.png", dpi=200)
    plt.close(fig)

    target_x = x_original[target_mask]
    target_x_repaired = x_repaired[target_mask]
    stacked = torch.cat([target_x, target_x_repaired], dim=0)
    stacked = stacked - stacked.mean(dim=0, keepdim=True)

    u, s, v = torch.pca_lowrank(stacked, q=2)
    proj = stacked @ v[:, :2]

    n = target_x.size(0)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(proj[:n, 0].cpu(), proj[:n, 1].cpu(), s=10, alpha=0.6, label="original")
    ax.scatter(proj[n:, 0].cpu(), proj[n:, 1].cpu(), s=10, alpha=0.6, label="repaired")
    ax.set_title("Target features: PCA before/after")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_pca_before_after.png", dpi=200)
    plt.close(fig)


def run_feature_repair(args: argparse.Namespace) -> None:
    graph_data = _load_graph(args.dataset)
    model = _load_model(args.checkpoint, args.trust_checkpoint)
    out_dir = _ensure_dir(args.plots_dir)

    cfg = FeatureRepairConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        max_target_nodes=args.max_target_nodes,
        l2_to_original=args.l2_to_original,
        use_entropy_weighting=not args.no_entropy_weighting,
    )

    if args.mode == "feature-repair":
        result = repair_features_overfit(model, graph_data, cfg)
        print("Baseline metrics:", result.metrics_base)
        if result.best_state is not None:
            print("Best metrics:", result.best_state["metrics"])
        target_mask = result.target_mask
    else:
        target_mask = _build_target_mask(model, graph_data, args.max_target_nodes)
        result = repair_features_all_nodes(model, graph_data, cfg, target_mask)
        print("Baseline metrics:", result.metrics_base)
        if result.best_state is not None:
            print("Best metrics:", result.best_state["metrics"])

    if out_dir is None or result.best_state is None:
        return

    device = next(model.parameters()).device
    graph_data = graph_data.to(device)
    x_original = graph_data.x.detach().clone()
    delta = result.best_state["delta"].to(device)
    target_mask_f = target_mask.float().unsqueeze(1).to(device)
    x_repaired = x_original + (delta * target_mask_f)

    graph_repaired = graph_data.clone()
    graph_repaired.x = x_repaired

    with torch.no_grad():
        logits_base, _ = model(graph_data)
        logits_repaired, _ = model(graph_repaired)
        pred_base = logits_base.argmax(dim=1).cpu()
        pred_repaired = logits_repaired.argmax(dim=1).cpu()

    _, correct_base = split_acc(
        pred_base,
        graph_data.y.cpu(),
        graph_data.train_mask.cpu(),
        graph_data.val_mask.cpu(),
        graph_data.test_mask.cpu(),
    )
    _, correct_repaired = split_acc(
        pred_repaired,
        graph_data.y.cpu(),
        graph_data.train_mask.cpu(),
        graph_data.val_mask.cpu(),
        graph_data.test_mask.cpu(),
    )

    node_id = _select_plot_node(target_mask.cpu(), correct_base, correct_repaired, args.target_node)
    _plot_neighborhood(
        graph_data.cpu(),
        pred_base,
        pred_repaired,
        node_id=node_id,
        hops=args.neighborhood_hops,
        out_dir=out_dir,
    )
    _plot_feature_drift(x_original.cpu(), x_repaired.cpu(), target_mask.cpu(), out_dir)


def run_gae_confidence(args: argparse.Namespace) -> None:
    graph_data = _load_graph(args.dataset)
    model = _load_model(args.checkpoint, args.trust_checkpoint)

    gae = train_gae(graph_data.x, graph_data.edge_index, GAEConfig())
    node_conf = compute_node_recon_confidence(gae, graph_data.x, graph_data.edge_index)

    with torch.no_grad():
        logits, _ = model(graph_data)
        preds = logits.argmax(dim=1)
    correct_mask = preds.eq(graph_data.y).cpu()
    node_conf = node_conf.cpu()

    print("Mean confidence (correct):", node_conf[correct_mask].mean().item())
    print("Mean confidence (incorrect):", node_conf[~correct_mask].mean().item())


def run_gae_khop_loss(args: argparse.Namespace) -> None:
    graph_data = _load_graph(args.dataset)
    model = _load_model(args.checkpoint, args.trust_checkpoint)

    gae = train_gae(graph_data.x, graph_data.edge_index, GAEConfig())

    with torch.no_grad():
        logits, _ = model(graph_data)
        preds = logits.argmax(dim=1)
    incorrect_nodes = torch.where(~preds.eq(graph_data.y))[0]

    loss = gae_k_hop_recon_loss(
        gae,
        graph_data.x,
        graph_data.edge_index,
        incorrect_nodes,
        k_hops=args.k_hops,
    )
    print(f"GAE {args.k_hops}-hop reconstruction loss: {loss:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run analysis utilities.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., ENGB, Cora)")
    parser.add_argument("--checkpoint", required=True, help="Path to a MyModel checkpoint")
    parser.add_argument(
        "--mode",
        choices=["feature-repair", "feature-repair-all", "gae-confidence", "gae-khop-loss"],
        default="feature-repair",
    )

    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-target-nodes", type=int, default=None)
    parser.add_argument("--l2-to-original", type=float, default=0.0)
    parser.add_argument("--no-entropy-weighting", action="store_true")

    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--target-node", type=int, default=None, help="Node id for neighborhood plots")
    parser.add_argument("--neighborhood-hops", type=int, default=1)

    parser.add_argument("--k-hops", type=int, default=1)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow non-weight objects when loading a checkpoint",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode in {"feature-repair", "feature-repair-all"}:
        run_feature_repair(args)
        return
    if args.mode == "gae-confidence":
        run_gae_confidence(args)
        return
    if args.mode == "gae-khop-loss":
        run_gae_khop_loss(args)
        return


if __name__ == "__main__":
    main()
