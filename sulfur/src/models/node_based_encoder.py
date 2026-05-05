import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class NodeBasedEncoder(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        diffusion_type: str,
        layers: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.K = K

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        # FIX: Explicitly configure GAT with heads and internal dropout
        self.gat = gnn.GAT(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            num_layers=layers,
            out_channels=self.hidden_dim,
            dropout=self.dropout_rate,  # Crucial for attention dropout
            heads=8,  # Multi-head attention stabilizes learning
            add_self_loops=False,  # PyG usually handles this, but good to be explicit if edge_index already has them
        )

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def reset_parameters(self):
        # FIX: If your training loop calls this to reset weights between folds/seeds,
        # returning early will cause data leakage between runs.
        self.encoder.reset_parameters()
        self.gat.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. Feature Encoding + Activation + Dropout
        x = self.encoder(x)
        x = F.elu(x)  # FIX: Added non-linearity
        x = self.dropout(x)

        # 2. GAT Message Passing
        # GAT compute its own attention weights with internal dropout applied
        x_gat = self.gat(x, edge_index)

        # 3. Residual Connection
        x = x_gat + x

        # 4. Final Activation + Dropout before Classification
        x = F.elu(x)  # FIX: Added non-linearity before decoder
        x = self.dropout(x)

        # 5. Decoding
        out = self.decoder(x)

        return out, torch.tensor(0)
