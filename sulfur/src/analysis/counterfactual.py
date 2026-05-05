from __future__ import annotations

from typing import List, Dict, Optional

import copy
import torch
from torch_geometric.data import Data


def remove_edge_both_directions(edge_index: torch.Tensor, u: int, v: int) -> torch.Tensor:
    """Remove an undirected edge (u, v) from a directed edge_index."""
    src, dst = edge_index
    keep = ~(((src == u) & (dst == v)) | ((src == v) & (dst == u)))
    return edge_index[:, keep]


def find_counterfactual_witnesses(
    model: torch.nn.Module,
    graph_data: Data,
    wrong_nodes: torch.Tensor,
    ground_truth: torch.Tensor,
    predictions: torch.Tensor,
    probs: torch.Tensor,
    ae_node_err: torch.Tensor,
    graph_nx: "object",
    top_neighbors_to_test: int = 5,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float | int]]:
    """Find edges whose removal flips an incorrect prediction to correct.

    The search tests the highest AE-error neighbors first.
    """
    device = device or next(model.parameters()).device
    model.eval()

    witnesses: List[Dict[str, float | int]] = []
    with torch.no_grad():
        for u_t in wrong_nodes:
            u = int(u_t.item())
            true_u = int(ground_truth[u].item())
            pred_u = int(predictions[u].item())

            neigh = list(graph_nx.neighbors(u))
            if not neigh:
                continue

            neigh_t = torch.tensor(neigh, dtype=torch.long)
            neigh_order = torch.argsort(ae_node_err[neigh_t], descending=True)
            candidates = neigh_t[neigh_order[:top_neighbors_to_test]].tolist()

            for v in candidates:
                new_ei = remove_edge_both_directions(graph_data.edge_index, u, v)
                cf_data = copy.deepcopy(graph_data)
                cf_data.edge_index = new_ei.to(device)
                cf_data.x = graph_data.x.to(device)

                logits_cf, _ = model(cf_data)
                logits_cf = logits_cf.cpu()
                pred_cf_u = int(logits_cf[u].argmax().item())
                fixed = pred_cf_u == true_u

                if fixed:
                    p_true_before = probs[u, true_u].item()
                    p_true_after = torch.softmax(logits_cf[u], dim=0)[true_u].item()
                    witnesses.append(
                        {
                            "u": u,
                            "v": int(v),
                            "true_label": true_u,
                            "pred_before": pred_u,
                            "pred_after": pred_cf_u,
                            "p_true_before": float(p_true_before),
                            "p_true_after": float(p_true_after),
                        }
                    )
                    break

    return witnesses
