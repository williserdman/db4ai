import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional
import torch_geometric.nn as gnn


class NodeBasedEncoder(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.K = K

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.gcns = nn.ModuleList(
            [
                gnn.GCNConv(self.hidden_dim, self.hidden_dim, improved=True)
                for _ in range(K)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(K)])
        self.lrs = nn.ModuleList([nn.LeakyReLU() for _ in range(K)])

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.decoder = nn.Linear(self.hidden_dim, network_info.num_classes)

        self.dropout = nn.Dropout(self.dropout_rate)

    def reset_parameters(self):
        return

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

        x = self.encoder(x)

        for i in range(self.K):
            xp = x.clone()
            xp = self.gcns[i](xp, edge_index)
            xp = self.bns[i](xp)
            xp = self.lrs[i](xp)
            x = self.dropout(xp + x)

        out = self.decoder(self.mlp(x))

        return F.log_softmax(out, dim=-1), torch.tensor(
            0,
        )
