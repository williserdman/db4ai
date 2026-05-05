from __future__ import annotations

import argparse
import csv
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
from .reconstruction import (
    AutoencoderConfig,
    GAEConfig,
    compute_node_ae_error,
    compute_node_recon_confidence,
    gae_k_hop_recon_loss,
    train_autoencoder,
    train_gae,
)
from .argon_adapter import load_argon_graph
from .text_embeddings import (
    BowConfig,
    build_text_index,
    embed_texts,
    update_embeddings_for_nodes,
)


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


def _plot_text_pca(
    emb_before: torch.Tensor,
    emb_after: torch.Tensor,
    node_indices: list[int],
    out_dir: Path,
) -> None:
    if emb_before.numel() == 0:
        return

    stacked = torch.cat([emb_before, emb_after], dim=0)
    stacked = stacked - stacked.mean(dim=0, keepdim=True)
    u, s, v = torch.pca_lowrank(stacked, q=2)
    proj = stacked @ v[:, :2]
    n = emb_before.size(0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(proj[:n, 0].cpu(), proj[:n, 1].cpu(), s=14, alpha=0.7, label="before")
    ax.scatter(proj[n:, 0].cpu(), proj[n:, 1].cpu(), s=14, alpha=0.7, label="after")

    for i, node_id in enumerate(node_indices):
        x0, y0 = proj[i, 0].item(), proj[i, 1].item()
        x1, y1 = proj[n + i, 0].item(), proj[n + i, 1].item()
        ax.plot([x0, x1], [y0, y1], color="tab:gray", alpha=0.4, linewidth=0.8)

    ax.set_title("Text embedding PCA (corrected nodes)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "text_embedding_pca_before_after.png", dpi=200)
    plt.close(fig)


def _rank_candidates(scores: torch.Tensor, higher_is_worse: bool, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= scores.numel():
        return torch.arange(scores.numel())
    if higher_is_worse:
        return torch.topk(scores, k=top_k).indices
    return torch.topk(-scores, k=top_k).indices


def _collect_neighbors(edge_index: torch.Tensor, target_indices: list[int]) -> dict[int, list[int]]:
    neighbors: dict[int, set[int]] = {idx: set() for idx in target_indices}
    edge_list = edge_index.t().tolist()
    for src, dst in edge_list:
        if src in neighbors:
            neighbors[src].add(dst)
        if dst in neighbors:
            neighbors[dst].add(src)
    return {idx: sorted(neigh) for idx, neigh in neighbors.items()}


def _export_candidates_csv(
    path: str,
    node_indices: list[int],
    node_ids: list[str],
    cascade_ids: list[str],
    node_texts: list[str],
    has_text: torch.Tensor,
    scores: torch.Tensor,
    edge_index: torch.Tensor,
    nearest_lookup: dict[int, tuple[str, float, str]],
) -> None:
    neighbors = _collect_neighbors(edge_index, node_indices)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_index",
                "node_id",
                "cascade_id",
                "has_text",
                "recon_score",
                "text",
                "nearest_node_id",
                "nearest_score",
                "nearest_text",
                "neighbor_ids",
            ],
        )
        writer.writeheader()
        for idx in node_indices:
            nearest = nearest_lookup.get(idx, ("", 0.0, ""))
            writer.writerow(
                {
                    "node_index": idx,
                    "node_id": node_ids[idx],
                    "cascade_id": cascade_ids[idx],
                    "has_text": bool(has_text[idx].item()),
                    "recon_score": float(scores[idx].item()),
                    "text": node_texts[idx],
                    "nearest_node_id": nearest[0],
                    "nearest_score": nearest[1],
                    "nearest_text": nearest[2],
                    "neighbor_ids": ";".join(str(node_ids[n]) for n in neighbors.get(idx, [])),
                }
            )


def _load_text_corrections(path: str, node_ids: list[str]) -> dict[int, str]:
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    corrections: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "text" not in row or not row["text"]:
                continue
            if "node_index" in row and row["node_index"]:
                idx = int(row["node_index"])
            elif "node_id" in row and row["node_id"]:
                idx = node_id_to_idx.get(row["node_id"], None)
                if idx is None:
                    continue
            else:
                continue
            corrections[idx] = row["text"]
    return corrections


def run_feature_repair(args: argparse.Namespace) -> None:
    if args.dataset is None or args.checkpoint is None:
        raise RuntimeError("Feature repair requires --dataset and --checkpoint.")
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


def run_argon_text_repair(args: argparse.Namespace) -> None:
    out_dir = _ensure_dir(args.plots_dir)
    if args.argon_edge_path is None or args.argon_text_path is None:
        raise RuntimeError("Argon mode requires --argon-edge-path and --argon-text-path.")
    argon = load_argon_graph(
        args.argon_edge_path,
        args.argon_text_path,
        drop_missing_text=args.argon_drop_missing_text,
        max_nodes=args.argon_max_nodes,
    )

    node_texts = argon.node_texts
    if not any(node_texts):
        raise RuntimeError("Argon graph has no non-empty texts to embed.")

    bow_config = BowConfig(
        max_features=args.text_max_features,
        min_df=args.text_min_df,
        lowercase=not args.text_no_lowercase,
        normalize=not args.text_no_normalize,
    )
    embed_result = embed_texts(
        node_texts,
        model_name=args.text_embedding_model,
        vocab_path=args.text_vocab_path,
        bow_config=bow_config,
    )
    embeddings = embed_result.embeddings

    graph_data = argon.data
    base_x = graph_data.x
    if args.text_concat and base_x is not None and base_x.numel() > 0:
        graph_data.x = torch.cat([base_x, embeddings], dim=1)
    else:
        graph_data.x = embeddings

    if args.recon_method == "gae":
        cfg = GAEConfig(
            hidden_channels=args.recon_hidden_channels,
            out_channels=args.recon_out_channels,
            lr=args.recon_lr,
            max_epochs=args.recon_epochs,
            log_every=args.recon_log_every,
        )
        recon_model = train_gae(graph_data.x, graph_data.edge_index, cfg)
        scores = compute_node_recon_confidence(recon_model, graph_data.x, graph_data.edge_index)
        higher_is_worse = False
    else:
        cfg = AutoencoderConfig(
            hidden_channels=args.recon_hidden_channels,
            out_channels=args.recon_out_channels,
            lr=args.recon_lr,
            max_epochs=args.recon_epochs,
            log_every=args.recon_log_every,
        )
        recon_model = train_autoencoder(graph_data.x, cfg)
        scores = compute_node_ae_error(recon_model, graph_data.x)
        higher_is_worse = True

    candidate_indices = _rank_candidates(scores, higher_is_worse, args.recon_top_k)
    candidate_list = [int(idx.item()) for idx in candidate_indices]

    nearest_lookup: dict[int, tuple[str, float, str]] = {}
    if argon.has_text.any():
        text_ids = [nid for nid, has_text in zip(argon.node_ids, argon.has_text) if has_text]
        text_values = [text for text, has_text in zip(argon.node_texts, argon.has_text) if has_text]
        text_embeddings = embeddings[argon.has_text]
        index = build_text_index(text_embeddings, text_ids, text_values, use_faiss=args.use_faiss)
        nearest = index.query(embeddings[candidate_indices].cpu().numpy(), k=1)
        for idx, nearest_row in zip(candidate_list, nearest):
            if nearest_row:
                nid, score, text = nearest_row[0]
                nearest_lookup[idx] = (nid, score, text)

    if args.export_csv:
        _export_candidates_csv(
            args.export_csv,
            candidate_list,
            argon.node_ids,
            argon.cascade_ids,
            argon.node_texts,
            argon.has_text,
            scores,
            graph_data.edge_index,
            nearest_lookup,
        )
        print(f"Exported {len(candidate_list)} candidates to {args.export_csv}")

    if not args.import_csv:
        return

    corrections = _load_text_corrections(args.import_csv, argon.node_ids)
    if not corrections:
        print("No text corrections found in import CSV.")
        return

    updated_texts = list(argon.node_texts)
    for idx, new_text in corrections.items():
        updated_texts[idx] = new_text

    before_embeddings = embeddings[torch.tensor(list(corrections.keys()), dtype=torch.long)]
    embeddings = update_embeddings_for_nodes(
        embeddings,
        list(corrections.keys()),
        [updated_texts[idx] for idx in corrections.keys()],
        embed_result.embedder,
    )
    if args.text_concat and base_x is not None and base_x.numel() > 0:
        graph_data.x = torch.cat([base_x, embeddings], dim=1)
    else:
        graph_data.x = embeddings

    if args.recon_method == "gae":
        recon_model = train_gae(graph_data.x, graph_data.edge_index, cfg)
        scores_after = compute_node_recon_confidence(recon_model, graph_data.x, graph_data.edge_index)
    else:
        recon_model = train_autoencoder(graph_data.x, cfg)
        scores_after = compute_node_ae_error(recon_model, graph_data.x)

    after_embeddings = embeddings[torch.tensor(list(corrections.keys()), dtype=torch.long)]
    if out_dir is not None:
        _plot_text_pca(before_embeddings.cpu(), after_embeddings.cpu(), list(corrections.keys()), out_dir)

    if args.recon_method == "gae":
        improvement = scores_after - scores
        label = "confidence"
    else:
        improvement = scores - scores_after
        label = "error"
    avg_improvement = improvement[torch.tensor(list(corrections.keys()), dtype=torch.long)].mean().item()
    print(f"Average reconstruction {label} improvement (corrected nodes): {avg_improvement:.4f}")


def run_gae_confidence(args: argparse.Namespace) -> None:
    if args.dataset is None or args.checkpoint is None:
        raise RuntimeError("GAE confidence requires --dataset and --checkpoint.")
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
    if args.dataset is None or args.checkpoint is None:
        raise RuntimeError("GAE k-hop loss requires --dataset and --checkpoint.")
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
    parser.add_argument("--dataset", required=False, help="Dataset name (e.g., ENGB, Cora)")
    parser.add_argument("--checkpoint", required=False, help="Path to a MyModel checkpoint")
    parser.add_argument(
        "--mode",
        choices=[
            "feature-repair",
            "feature-repair-all",
            "gae-confidence",
            "gae-khop-loss",
            "argon-text-repair",
        ],
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

    parser.add_argument("--argon-edge-path", type=str, default=None, help="Argon edge file path")
    parser.add_argument("--argon-text-path", type=str, default=None, help="Argon text TSV path")
    parser.add_argument("--argon-drop-missing-text", action="store_true")
    parser.add_argument("--argon-max-nodes", type=int, default=None)

    parser.add_argument("--text-embedding-model", type=str, default="bow")
    parser.add_argument("--text-vocab-path", type=str, default=None)
    parser.add_argument("--text-max-features", type=int, default=20000)
    parser.add_argument("--text-min-df", type=int, default=1)
    parser.add_argument("--text-no-lowercase", action="store_true")
    parser.add_argument("--text-no-normalize", action="store_true")
    parser.add_argument("--text-concat", action="store_true")
    parser.add_argument("--use-faiss", action="store_true")

    parser.add_argument("--recon-method", choices=["gae", "ae"], default="gae")
    parser.add_argument("--recon-epochs", type=int, default=400)
    parser.add_argument("--recon-log-every", type=int, default=50)
    parser.add_argument("--recon-hidden-channels", type=int, default=128)
    parser.add_argument("--recon-out-channels", type=int, default=64)
    parser.add_argument("--recon-lr", type=float, default=0.01)
    parser.add_argument("--recon-top-k", type=int, default=50)
    parser.add_argument("--export-csv", type=str, default=None)
    parser.add_argument("--import-csv", type=str, default=None)

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
    if args.mode == "argon-text-repair":
        run_argon_text_repair(args)
        return


if __name__ == "__main__":
    main()
