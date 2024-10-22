# -*- coding: utf-8 -*-

# @Time  : 2022/9/27 15:12
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import torch
import torch.nn.functional as F
from math import pi as PI
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, cutoff):
        super().__init__(aggr='mean')
        self.lin1 = Linear(in_channels, in_channels, bias=False)
        self.lin2 = Linear(in_channels, out_channels)
        self.nn = nn
        self.act = ShiftedSoftplus()
        self.cutoff = cutoff
        self.lin3 = Linear(out_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        C = (torch.cos(edge_weight * PI / self.cutoff) + 1.0) * 0.5
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)

        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CGGRU(torch.nn.Module):
    def __init__(self, nfeat_node=19, nc_edge_hidden=50, dim=128, cutoff=5.0, **kwargs):
        super().__init__()
        """nfeat_edge=None, nfeat_state=0"""

        self.lin0 = torch.nn.Linear(nfeat_node, dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, nc_edge_hidden)

        nn = Sequential(Linear(nc_edge_hidden, 128), ReLU(), Linear(128, dim))
        self.conv = CFConv(dim, dim, nn, cutoff)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        data.edge_attr = self.distance_expansion(data.edge_weight)

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr, data.edge_weight))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)
