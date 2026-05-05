import os

import torch
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Actor,
    WebKB,
    HeterophilousGraphDataset,
)

from torch_geometric.transforms import NormalizeFeatures
import collections
import numpy as np
import torch_geometric.data as tg_data
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, cast


import json
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

import os
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data


class DownloadedTwitchDataset(InMemoryDataset):
    def __init__(self, root, name: str, transform=None, pre_transform=None):
        self.name = name
        assert name in {"DE", "ENGB", "ES", "FR", "PTBR", "RU"}

        super().__init__(root, transform, pre_transform)

        # FIX 1: Modern PyG data loading
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            f"musae_{self.name}_edges.csv",
            f"musae_{self.name}_target.csv",
            f"musae_{self.name}_features.json",
        ]

    @property
    def processed_file_names(self):
        return [f"data_{self.name}.pt"]

    @property
    def num_classes(self):
        if self.data.y is None or self.data.y.numel() == 0:
            return 0
        return int(self.data.y.max()) + 1

    # FIX 3: Removed redundant num_classes and num_features properties
    # (InMemoryDataset handles these automatically)

    def process(self):
        edges_path = os.path.join(self.raw_dir, f"musae_{self.name}_edges.csv")
        target_path = os.path.join(self.raw_dir, f"musae_{self.name}_target.csv")
        features_path = os.path.join(self.raw_dir, f"musae_{self.name}_features.json")

        edges_df = pd.read_csv(edges_path)
        target_df = pd.read_csv(target_path)

        with open(features_path, "r") as f:
            features_dict = json.load(f)

        # Make sure the target IDs are integers
        target_df["new_id"] = target_df["new_id"].astype(int)

        # JSON keys are always strings, convert them to ints
        features_dict = {int(k): v for k, v in features_dict.items()}

        feature_nodes = set(features_dict.keys())

        # FIX: Use new_id instead of iloc[:, 0]
        target_nodes = set(target_df["new_id"])

        # Take intersection
        node_ids = sorted(feature_nodes & target_nodes)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # -------- Edge index --------
        edges_df["from"] = edges_df["from"].astype(int)
        edges_df["to"] = edges_df["to"].astype(int)

        src = edges_df["from"].map(node_id_to_idx)
        dst = edges_df["to"].map(node_id_to_idx)

        # Filter out edges with missing nodes (NaN values)
        valid_mask = src.notna() & dst.notna()
        src = src[valid_mask].astype(int)
        dst = dst[valid_mask].astype(int)

        # FIX 4: Wrap in np.array to avoid PyTorch slow execution warnings
        edge_index = torch.tensor(np.array([src.values, dst.values]), dtype=torch.long)

        # -------- Build feature matrix --------
        num_nodes = len(node_ids)

        # Find total number of features across all nodes
        num_features = (
            max(max(feats, default=-1) for feats in features_dict.values()) + 1
        )

        x = torch.zeros((num_nodes, num_features), dtype=torch.float)

        for i, nid in enumerate(node_ids):
            feat_indices = features_dict[nid]
            if feat_indices:
                x[i, feat_indices] = 1.0

        # -------- Targets --------
        # FIX: Map from 'new_id' to 'mature' (casting to int handles True/False booleans safely)
        target_map = dict(zip(target_df["new_id"], target_df["mature"].astype(int)))

        y = torch.tensor([target_map[nid] for nid in node_ids], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save properly
        torch.save(self.collate([data]), self.processed_paths[0])


ALL_DATASETS = [
    # "Questions",
    # "Cora",
    # "Roman-empire",
    # "computers",
    # "photo",
    # "Citeseer",
    # "Pubmed",
    # "squirrel",
    # "chameleon",
    # "actor",
    # "texas",
    # "cornell",
    # "Amazon-ratings",
    # "Minesweeper",
    # "Tolokers",
    "ENGB",
]


def _load_filtered_dataset(path):
    # Load the .npz file
    data = np.load(path)
    num_nodes, nf = data["node_features"].shape
    nc = len(np.unique(data["node_labels"]))

    # Convert to PyTorch tensors
    x = torch.tensor(data["node_features"], dtype=torch.float)
    y = torch.tensor(data["node_labels"], dtype=torch.long)
    edge_index = torch.tensor(data["edges"], dtype=torch.long).t().contiguous()

    # Load the 10 fixed splits (masks)
    # The file usually contains 'train_masks', 'val_masks', 'test_masks'
    # Shape: [num_nodes, 10]
    train_masks = torch.tensor(data["train_masks"], dtype=torch.bool)
    val_masks = torch.tensor(data["val_masks"], dtype=torch.bool)
    test_masks = torch.tensor(data["test_masks"], dtype=torch.bool)

    return (
        tg_data.Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_masks,
            val_mask=val_masks,
            test_mask=test_masks,
        ),
        nf,
        nc,
    )


class LightningGraph:
    def __init__(self, data, num_features, num_classes, class_weights):
        self.data = data
        self.num_features = num_features
        self.num_classes = num_classes
        self.class_weights = class_weights


def _create_ds_splits(name, data: tg_data.Data, train_split, val_split, test_split):
    # Create new train/val/test masks based on provided splits

    num_nodes = data.y.size(0)  # type: ignore
    indices = torch.randperm(num_nodes)
    train_end = int(train_split * num_nodes)
    val_end = train_end + int(val_split * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    """ print(
        f"{name}: train={int(data.train_mask.sum())}, val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())}"
    ) """
    return data


@dataclass
class SimpleDatasetWrapper:
    def __init__(self):
        self.items = []
        self.num_features: Optional[int] = None
        self.num_classes: Optional[int] = None

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, item):
        self.items.append(item)

    def __len__(self):
        return len(self.items)


def _load_single_ds(name: str):
    name = name.strip()
    tfm = (
        NormalizeFeatures()
    )  # creating instance of NormalizeFeatures to pass into transform (each row sums to one)

    if name in {"Cora", "Citeseer", "Pubmed"}:
        ds = Planetoid(
            root=os.path.join("data", name), name=name, transform=tfm
        )  # initialize planetoid dataset object, will download or use downloaded copy

    elif name in {"chameleon", "squirrel"}:
        data, nf, nc = _load_filtered_dataset(
            Path(f"data/{name}/{name}_filtered_directed.npz")
        )

        ds = SimpleDatasetWrapper()

        ds.append(data)
        ds.num_features = nf
        ds.num_classes = nc

    elif name in {"computers", "photo"}:
        ds = Amazon(root=os.path.join("data", name), name=name, transform=tfm)
        data = cast(tg_data.Data, ds[0])
        nf = ds.num_features
        nc = ds.num_classes
        ds = SimpleDatasetWrapper()
        for _ in range(10):
            split_data = _create_ds_splits(name, data, 0.6, 0.2, 0.2)
            ds.append(split_data)
        ds.num_features, ds.num_classes = nf, nc

    elif name in {"actor"}:
        ds = Actor(root=os.path.join("data", name), transform=tfm)

    elif name in {"texas", "cornell"}:
        ds = WebKB(root=os.path.join("data", name), name=name, transform=tfm)

    elif name in {
        "Roman-empire",
        "Amazon-ratings",
        "Minesweeper",
        "Tolokers",
        "Questions",
    }:
        ds = HeterophilousGraphDataset(
            root=os.path.join("data", name), name=name, transform=tfm
        )

    elif name in {"ENGB"}:

        ds = DownloadedTwitchDataset(
            os.path.join("data", "twitch", name), name=name, transform=tfm
        )
        data = cast(tg_data.Data, ds[0])
        nf = ds.num_features
        nc = ds.num_classes
        ds = SimpleDatasetWrapper()
        for _ in range(10):
            split_data = _create_ds_splits(name, data, 0.6, 0.2, 0.2)
            ds.append(split_data)
        ds.num_features, ds.num_classes = nf, nc

    else:
        print(f"dataset {name} not found, continuing")
        return None

    SPLIT_INDEX = 1
    data = ds[0]

    if name in {
        "Roman-empire",
        "actor",
        "texas",
        "cornell",
        "Amazon-ratings",
        "Minesweeper",
        "Tolokers",
        "Questions",
    }:
        data.train_mask = data.train_mask[:, SPLIT_INDEX]  # type: ignore
        data.val_mask = data.val_mask[:, SPLIT_INDEX]  # type: ignore
        data.test_mask = data.test_mask[:, SPLIT_INDEX]  # type: ignore

    elif name in {
        "squirrel",
        "chameleon",
    }:
        data.train_mask = data.train_mask[SPLIT_INDEX, :]  # type: ignore
        data.val_mask = data.val_mask[SPLIT_INDEX, :]  # type: ignore
        data.test_mask = data.test_mask[SPLIT_INDEX, :]  # type: ignore

    elif name in {"computers", "photo"}:
        data = ds[SPLIT_INDEX]

    # print(ds.num_features)
    print(data)
    return (
        data,
        ds.num_features,
        ds.num_classes,
    )


def load_datasets(datasets: list[str]) -> dict[str, LightningGraph]:

    # we want to grab a "Data" object (torch geometric concept) for each dataset. a data object is one graph. some of the datasets have more than one graph
    out = collections.defaultdict(None)
    for n in datasets:
        ds, nf, nc = _load_single_ds(n)  # type: ignore
        # compute class frequencies from labels
        y = ds.y  # type: ignore
        if y.dim() > 1:
            y = y.view(-1)
        y = y.long()
        counts = torch.bincount(y, minlength=nc).float()  # type: ignore
        # avoid divide-by-zero for classes with no samples
        counts = counts + 1e-8
        class_frequencies = counts / counts.sum()
        # Inverse frequencies (or other weighting scheme)
        class_weights = 1.0 / class_frequencies
        # Normalize or scale if desired, e.g., to sum to 1
        class_weights = class_weights / class_weights.sum()

        # Ensure the weight tensor is of type float
        class_weights = class_weights.float()
        t = tg_data.lightning.LightningNodeData(ds, loader="full")
        out[n] = LightningGraph(t, nf, nc, torch.tensor(class_weights))
        # print(out[n].num_features)
    return out


if __name__ == "__main__":
    load_datasets(ALL_DATASETS)
