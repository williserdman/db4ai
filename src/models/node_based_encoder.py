import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
from torch_geometric.utils import (
    to_scipy_sparse_matrix,
    to_dense_adj,
    dense_to_sparse,
    add_remaining_self_loops,
)
import scipy.sparse as sp
from torch_geometric.nn.inits import zeros


from typing import Callable, Optional, Union
import torch as th
from torch.nn.modules.module import Module, _grad_t
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

# from utils import cheby, init_temp


class DiffusionStep(MessagePassing, nn.Module):
    """
    Message passing layer for Graph Diffusion based attention.
    Performs X' = P @ X where P is the proppagation matrix.
    """

    def __init__(self, prop_type: str, K: int):
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
            out = [self.coeffs[0] * x]
            curr_x = x

            for i in range(1, self.K + 1):
                curr_x = self.propagate(edge_index, x=curr_x, edge_weight=edge_weight)
                out.append(self.coeffs[i] * curr_x)

            return out

        elif self.prop_type == "chebyshev":
            # chebyshev
            # T_0 = x
            # T_1 = L x = x - A_norm x
            # T_k = 2 * L * T_{k-1} - T_{k-2}
            A_x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            L_x = x - A_x
            out = [self.coeffs[0] * x, self.coeffs[1] * L_x]

            if self.K == 1:
                return out

            T_k_minus_two = x
            T_k_minus_one = L_x
            for k in range(2, self.K + 1):
                A_tm1 = self.propagate(
                    edge_index, x=T_k_minus_one, edge_weight=edge_weight
                )
                L_tm1 = T_k_minus_one - A_tm1
                T_k = 2.0 * L_tm1 - T_k_minus_two
                out.append(self.coeffs[k] * T_k)
                T_k_minus_two, T_k_minus_one = T_k_minus_one, T_k

            return out

    def message(self, x_j, edge_weight):  # type: ignore
        """
        Docstring for message

        :param self: Description
        :param x_j: features of the source node
        :param edge_weight: will be applied
        """

        return edge_weight.reshape(-1, 1) * x_j


class PolyAttn(nn.Module):
    def __init__(self, K, base, hidden_dim, n_head, multi, dropout_rate, q):
        super(PolyAttn, self).__init__()
        self.K = K + 1
        self.base = base
        self.norm = nn.LayerNorm(hidden_dim)
        self.n_head = n_head
        self.multi = multi
        self.d_head = hidden_dim // n_head

        self.token_wise_network = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, int(hidden_dim * self.multi)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim * self.multi), hidden_dim),
                )
                for _ in range(self.K)
            ]
        )

        self.W_Q = nn.Linear(hidden_dim, self.n_head * self.d_head, bias=False)
        self.W_K = nn.Linear(hidden_dim, self.n_head * self.d_head, bias=False)

        self.bias_scale = nn.Parameter(torch.ones(self.n_head, self.K))
        self.bias = torch.tensor([((j + 1) ** q) ** (-1) for j in range(self.K)])

        self.dprate = dropout_rate
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.token_wise_network:
            layer[0].reset_parameters()  # type: ignore
            layer[2].reset_parameters()  # type: ignore
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()

    def forward(self, src):
        batch_size = src.shape[0]
        origin_src = src
        src = self.norm(src)
        token = src
        value = src
        token = torch.stack(
            [
                layer(token[:, idx, :])
                for idx, layer in enumerate(self.token_wise_network)
            ],
            dim=1,
        )
        query = self.W_Q(token)
        key = self.W_K(token)
        q_heads = query.view(
            query.size(0), query.size(1), self.n_head, self.d_head
        ).transpose(
            1, 2
        )  # [n,n_head,k,d_head]
        k_heads = key.view(
            key.size(0), key.size(1), self.n_head, self.d_head
        ).transpose(1, 2)
        v_heads = value.view(value.size(0), value.size(1), self.n_head, -1).transpose(
            1, 2
        )
        attention_scores = torch.matmul(
            q_heads, k_heads.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(self.d_head).float())
        attention_scores = torch.tanh(attention_scores)
        attn_mask = torch.einsum("hk,k->hk", self.bias_scale, self.bias.to(src.device))
        attention_scores = torch.einsum("nhdk,hk->nhdk", attention_scores, attn_mask)
        attention_scores = F.dropout(
            attention_scores, p=self.dprate, training=self.training
        )
        context_heads = torch.matmul(attention_scores, v_heads)
        context_sequence = (
            context_heads.transpose(1, 2).contiguous().view(batch_size, self.K, -1)
        )
        src = F.dropout(context_sequence, p=self.dprate, training=self.training)
        src = src + origin_src
        return src


class FFNNetwork(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super(FFNNetwork, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(ffn_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x


class FFN(nn.Module):
    def __init__(self, K, base, dropout_rate, hidden_dim, ffn_dim):
        super(FFN, self).__init__()
        self.K = K + 1
        self.base = base
        self.dropout = dropout_rate
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_net = FFNNetwork(hidden_dim, ffn_dim)

    def forward(self, src):
        origin_src = src
        src = self.ffn_norm(src)
        src = self.ffn_net(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = src + origin_src
        return src


class PolyFormerBlock(nn.Module):
    def __init__(self, K, base, hidden_dim, n_head, multi, dropout_rate, q, ffn_dim):
        super(PolyFormerBlock, self).__init__()
        self.K = K + 1
        self.base = base

        self.attnmodule = PolyAttn(K, base, hidden_dim, n_head, multi, dropout_rate, q)
        self.ffnmodule = FFN(K, base, dropout_rate, hidden_dim, ffn_dim)

    def forward(self, src):
        src = self.attnmodule(src)
        src = self.ffnmodule(src)
        return src


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


class PolyFormer(nn.Module):
    def __init__(
        self,
        network_info,
        hidden_dim: int,
        dropout_rate: int,
        num_layers: int,
        K: int,
        base: str,
        n_head: int,
        multi: int,
        q: float,
        ffn_dim: int,
    ):
        super(PolyFormer, self).__init__()
        self.dropout = dropout_rate
        self.nlayers = num_layers
        self.hidden_dim = hidden_dim
        self.attn = nn.ModuleList(
            [
                PolyFormerBlock(
                    K, base, hidden_dim, n_head, multi, dropout_rate, q, ffn_dim
                )
                for _ in range(self.nlayers)
            ]
        )
        self.K = K + 1
        self.base = base

        self.lin1 = Linear(network_info.num_features, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim)
        self.lin3 = Linear(hidden_dim, network_info.num_classes)

        self.diffusion = DiffusionStep(base, K)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        N, C = x.shape

        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=N, add_self_loops=True, dtype=x.dtype
        )
        input_mat = self.diffusion.forward(x, edge_index, edge_weight)
        input_mat = torch.stack(input_mat, dim=1)  # [N,k,d]

        input_mat = self.lin1(input_mat)  # just for common dataset

        for block in self.attn:
            input_mat = block(input_mat)

        x = torch.sum(input_mat, dim=1)  # [N,d]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return x, torch.tensor(0)
