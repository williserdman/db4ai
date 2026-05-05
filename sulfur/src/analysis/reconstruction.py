from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric import nn as gnn
from torch_geometric.utils import degree, k_hop_subgraph, subgraph


@dataclass
class AutoencoderConfig:
    hidden_channels: int = 128
    out_channels: int = 64
    lr: float = 0.01
    max_epochs: int = 4000
    lr_factor: float = 0.5
    patience: int = 200
    log_every: int = 50


@dataclass
class GAEConfig:
    hidden_channels: int = 128
    out_channels: int = 64
    lr: float = 0.01
    max_epochs: int = 4000
    lr_factor: float = 0.5
    patience: int = 200
    log_every: int = 50


class StandardAutoencoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def train_autoencoder(
    x: torch.Tensor,
    config: AutoencoderConfig,
    device: Optional[torch.device] = None,
) -> StandardAutoencoder:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandardAutoencoder(x.size(1), config.hidden_channels, config.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.lr_factor, patience=config.patience)
    criterion = nn.MSELoss()

    x = x.to(device)
    model.train()
    for epoch in range(1, config.max_epochs + 1):
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % config.log_every == 0:
            print(f"Epoch: {epoch:04d}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model


def train_gae(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    config: GAEConfig,
    device: Optional[torch.device] = None,
) -> gnn.GAE:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gnn.GAE(GCNEncoder(x.size(1), config.hidden_channels, config.out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.lr_factor, patience=config.patience)

    x = x.to(device)
    edge_index = edge_index.to(device)

    model.train()
    for epoch in range(1, config.max_epochs + 1):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % config.log_every == 0:
            print(f"Epoch: {epoch:04d}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model


def compute_node_recon_confidence(
    model_gae: gnn.GAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or next(model_gae.parameters()).device
    model_gae.eval()
    with torch.no_grad():
        z = model_gae.encode(x.to(device), edge_index.to(device))
        edge_probs = model_gae.decoder(z, edge_index.to(device), sigmoid=True)
        node_error_sum = torch.zeros(x.size(0), device=device)
        node_error_sum.scatter_add_(0, edge_index[0].to(device), edge_probs)
        node_degree = degree(edge_index[0], num_nodes=x.size(0)).to(device)
        node_avg_conf = node_error_sum / (node_degree + 1e-6)
    return node_avg_conf


def compute_node_ae_error(
    model_ae: StandardAutoencoder,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or next(model_ae.parameters()).device
    model_ae.eval()
    with torch.no_grad():
        recon = model_ae(x.to(device))
        err = (recon - x.to(device)).pow(2).mean(dim=1)
    return err


def reconstruct_edges_by_threshold(
    edge_index: torch.Tensor,
    edge_probs: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    edge_mask = edge_probs > threshold
    return edge_index[:, edge_mask.cpu()]


def gae_k_hop_recon_loss(
    model_gae: gnn.GAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_indices: torch.Tensor,
    k_hops: int = 1,
    device: Optional[torch.device] = None,
) -> float:
    device = device or next(model_gae.parameters()).device
    model_gae.eval()

    subset, _, _, _ = k_hop_subgraph(node_indices, num_hops=k_hops, edge_index=edge_index)
    edge_index_k_hop, _ = subgraph(subset, edge_index, relabel_nodes=True)
    x_k_hop = x[subset].to(device)

    with torch.no_grad():
        z_k_hop = model_gae.encode(x_k_hop, edge_index_k_hop.to(device))
        recon_loss = model_gae.recon_loss(z_k_hop, edge_index_k_hop.to(device))
    return float(recon_loss.item())
