import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional


class NodeBasedEncoder(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        flatten_size: int,
    ):
        super().__init__()

        self.K = K  # args.K
        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.compress = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, flatten_size * 2),
            nn.LeakyReLU(),
            nn.LayerNorm(flatten_size * 2),
            nn.Linear(flatten_size * 2, flatten_size),
        )

        self.decompress = nn.Sequential(
            nn.LayerNorm(flatten_size),
            nn.Linear(flatten_size, flatten_size * 2),
            nn.LeakyReLU(),
            nn.LayerNorm(flatten_size * 2),
            nn.Linear(flatten_size * 2, self.hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.decoder = nn.Linear(self.hidden_dim, network_info.num_features)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.mse = nn.MSELoss()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        t = self.encoder(x)

        N, H = x.shape

        # Pre-compute adjacency
        # edge_index, edge_weight = gcn_norm(
        #     edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        # )
        # Laplacian
        # edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=N)  # type: ignore

        compressed = self.compress(t)
        decompressed = self.decompress(compressed)
        decoded = self.decoder(decompressed)

        reconstruction_loss = self.mse(decoded, x)

        return compressed, reconstruction_loss
