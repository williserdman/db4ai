import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from args import MyArgs
from torch_geometric.utils import get_laplacian
from typing import Optional
import torch_geometric.nn as gnn


class DiffusionStep(MessagePassing, nn.Module):
    """
    Message passing layer for Graph Diffusion based attention.
    Performs X' = P @ X where P is the proppagation matrix.
    """

    def __init__(self, prop_type: str, K: int, hidden_dim):
        super(DiffusionStep, self).__init__(aggr="sum")
        assert prop_type in {"monomial", "chebyshev", "mlp"}, prop_type

        self.prop_type = prop_type
        self.K = K

        self.coeffs = nn.Parameter(
            torch.tensor(torch.ones((K + 1)) * (1.0 / (K + 1)))
        )  # Initialize so sum is approx 1

    def forward(
        self, x, edge_index, edge_weight, old_info: Optional[torch.Tensor] = None
    ) -> list[torch.Tensor]:  # type: ignore
        """
        Docstring for forward

        :param self: Description
        :param x: (N, num_channels)
        :param edge_index: (2, num_edges)
        :param edge_weight: (E), computed through gcn_norm
        """

        out = [x]
        if self.K <= 0:
            return out

        if self.prop_type == "monomial":
            out_sum = self.coeffs[0] * x
            curr_x = x

            for i in range(1, self.K + 1):
                curr_x = self.propagate(edge_index, x=curr_x, edge_weight=edge_weight)
                out_sum = out_sum + self.coeffs[i] * curr_x

            return [out_sum]

        elif self.prop_type == "chebyshev":
            # chebyshev
            # T_0 = x
            # T_1 = L x = x - A_norm x
            # T_k = 2 * L * T_{k-1} - T_{k-2}
            A_x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            L_x = x - A_x
            out_sum = self.coeffs[0] * x + self.coeffs[1] * L_x

            if self.K == 1:
                return [out_sum]

            T_k_minus_two = x
            T_k_minus_one = L_x
            for k in range(2, self.K + 1):
                A_tm1 = self.propagate(
                    edge_index, x=T_k_minus_one, edge_weight=edge_weight
                )
                L_tm1 = T_k_minus_one - A_tm1
                T_k = 2.0 * L_tm1 - T_k_minus_two
                out_sum = out_sum + self.coeffs[k] * T_k
                T_k_minus_two, T_k_minus_one = T_k_minus_one, T_k

            return [out_sum]

    def message(self, x_j, edge_weight):  # type: ignore
        """
        Docstring for message

        :param self: Description
        :param x_j: features of the source node
        :param edge_weight: will be applied
        """

        return edge_weight.reshape(-1, 1) * x_j


class NodeBasedEncoder(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: float,
        K: int,
        diffusion_type: str,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = network_info.num_classes
        self.dropout_rate = dropout_rate
        self.K = K

        self.encoder = nn.Linear(network_info.num_features, self.hidden_dim)

        self.diff = DiffusionStep(diffusion_type, K, hidden_dim)
        self.use_lap = True if diffusion_type in {"chebyshev"} else False

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.decoder = nn.Linear(self.hidden_dim, network_info.num_classes)

        self.dropout = nn.Dropout(self.dropout_rate)

    def reset_parameters(self):
        return

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        N, feature_count = x.shape
        x = self.dropout(x)  # Apply dropout to input features
        x = self.encoder(x)
        x = self.dropout(x)  # Apply dropout after first layer
        original_x = x
        N, H = x.shape

        x = self.mlp(x) + original_x
        x = self.dropout(x)  # Apply dropout after MLP

        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )
        """ if self.use_lap:
            edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=N)  # type: ignore """

        msgs = self.diff.forward(x, edge_index, edge_weight)
        stack_msgs = torch.stack(msgs, dim=1)
        sum_msgs = torch.sum(stack_msgs, dim=1)

        out = self.decoder(self.dropout(self.mlp2(sum_msgs + original_x) + sum_msgs))

        return out, torch.tensor(
            0,
        )
