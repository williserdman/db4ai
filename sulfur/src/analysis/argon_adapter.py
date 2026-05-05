from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data


@dataclass
class ArgonGraph:
    data: Data
    node_ids: list[str]
    node_texts: list[str]
    has_text: torch.Tensor
    cascade_ids: list[str]


def load_argon_graph(
    edge_path: str | Path,
    text_path: str | Path,
    drop_missing_text: bool = False,
    max_nodes: Optional[int] = None,
) -> ArgonGraph:
    text_map = _load_text_map(text_path)

    node_ids: list[str] = []
    cascade_ids: list[str] = []
    node_id_to_idx: dict[str, int] = {}
    edges: list[tuple[str, str]] = []

    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            root_id, parent_str, node_id = parts[0], parts[1], parts[2]

            for nid in (node_id, parent_str):
                if nid == "None":
                    continue
                if nid not in node_id_to_idx:
                    node_id_to_idx[nid] = len(node_ids)
                    node_ids.append(nid)
                    cascade_ids.append(root_id)

            if parent_str != "None":
                edges.append((parent_str, node_id))

    node_texts = [text_map.get(nid, "") for nid in node_ids]
    has_text = torch.tensor([bool(text) for text in node_texts], dtype=torch.bool)

    if drop_missing_text:
        keep_ids = {nid for nid, text in zip(node_ids, node_texts) if text}
        node_ids, node_texts, cascade_ids, has_text = _filter_nodes(
            node_ids, node_texts, cascade_ids, has_text, keep_ids
        )
        edges = [(src, dst) for src, dst in edges if src in keep_ids and dst in keep_ids]

    if max_nodes is not None and max_nodes < len(node_ids):
        keep_ids = set(node_ids[:max_nodes])
        node_ids, node_texts, cascade_ids, has_text = _filter_nodes(
            node_ids, node_texts, cascade_ids, has_text, keep_ids
        )
        edges = [(src, dst) for src, dst in edges if src in keep_ids and dst in keep_ids]

    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    edge_index = torch.tensor(
        [[node_id_to_idx[src], node_id_to_idx[dst]] for src, dst in edges],
        dtype=torch.long,
    ).t()

    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x = torch.zeros((len(node_ids), 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return ArgonGraph(
        data=data,
        node_ids=node_ids,
        node_texts=node_texts,
        has_text=has_text,
        cascade_ids=cascade_ids,
    )


def _load_text_map(text_path: str | Path) -> dict[str, str]:
    text_map: dict[str, str] = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            text_map[parts[0]] = parts[1]
    return text_map


def _filter_nodes(
    node_ids: list[str],
    node_texts: list[str],
    cascade_ids: list[str],
    has_text: torch.Tensor,
    keep_ids: set[str],
) -> tuple[list[str], list[str], list[str], torch.Tensor]:
    keep_mask = [nid in keep_ids for nid in node_ids]
    filtered_ids = [nid for nid, keep in zip(node_ids, keep_mask) if keep]
    filtered_texts = [text for text, keep in zip(node_texts, keep_mask) if keep]
    filtered_cascades = [cid for cid, keep in zip(cascade_ids, keep_mask) if keep]
    filtered_has_text = has_text[torch.tensor(keep_mask, dtype=torch.bool)]
    return filtered_ids, filtered_texts, filtered_cascades, filtered_has_text
