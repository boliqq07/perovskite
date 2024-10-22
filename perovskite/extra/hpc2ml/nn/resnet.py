# -*- coding: utf-8 -*-

# @Time  : 2022/9/10 16:26
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
from torch import nn
from torch.nn import Module, ModuleList, BatchNorm1d

from hpc2ml.nn.templatemodel import get_active_layer


class ResBlockSameSize(Module):
    """

    ResN(nn.Linear,n_res=2,in_features=64,out_features=64)

    """

    def __init__(self, m, n_res=1, batch_norm=True, active="ReLU", force_x_match=False, **kwargs):
        super().__init__()

        res_layer = ModuleList()

        for _ in range(n_res):
            res_layer.append(m(**kwargs))
            if batch_norm is True:
                bm = BatchNorm1d(num_features=list(kwargs.values())[1])  # num_features = out_features
                res_layer.append(bm)
            res_layer.append(get_active_layer(active)())

        self.first_layer = res_layer[0]
        self.middle_layer = res_layer[1:-1]  # del first and last active layer
        self.force_x_match = force_x_match
        if force_x_match:
            self.x_match_layer = nn.Linear(list(kwargs.values())[0], list(kwargs.values())[1])
        self.last_active_layer = res_layer[-1]

        self.n_res = n_res
        self.block_size = 3 if batch_norm else 2

    def reset_parameters(self):
        self.first_layer.reset_parameters()
        self.middle_layer.reset_parameters()

    def forward(self, h, data=None):

        out = self.first_layer(h, data=data)

        for n, ci in enumerate(self.middle_layer):
            if (n + 1) % self.block_size == 0:
                out = ci(out, data=data)  # m.
            else:
                out = ci(out)  # active or relu.

        if self.force_x_match:
            h = self.x_match_layer(h)

        out = self.last_active_layer(out + h)

        return out


class ResBlockDiffSize(Module):
    """

    ResBlockDiffSize(nn.Linear,layer_size_seq=(64,128,256))

    """

    def __init__(self, m, layer_size_seq=(64, 128, 256), batch_norm=True, active="ReLU",
                 force_x_match=False, **kwargs):
        super().__init__()
        assert len(layer_size_seq) >= 2

        res_layer = ModuleList()

        for i in range(len(layer_size_seq) - 1):
            res_layer.append(m(layer_size_seq[i], layer_size_seq[i + 1], **kwargs))  # in,out, others.
            if batch_norm is True:
                bm = BatchNorm1d(num_features=layer_size_seq[i + 1])  # num_features = out_features
                res_layer.append(bm)
            res_layer.append(get_active_layer(active)())

        self.first_layer = res_layer[0]
        self.middle_layer = res_layer[1:-1]  # del first and last active layer
        self.force_x_match = force_x_match
        if force_x_match:
            self.x_match_layer = nn.Linear(layer_size_seq[0], layer_size_seq[-1])
        self.last_active_layer = res_layer[-1]

        self.n_res = len(layer_size_seq) - 1
        self.block_size = 3 if batch_norm else 2

    def reset_parameters(self):
        self.first_layer.reset_parameters()
        self.middle_layer.reset_parameters()

    def forward(self, h, data=None):

        out = self.first_layer(h, data=data)

        for n, ci in enumerate(self.middle_layer):
            if (n + 1) % self.block_size == 0:
                out = ci(out, data=data)  # m.
            else:
                out = ci(out)  # active or relu.

        if self.force_x_match:
            h = self.x_match_layer(h)

        out = self.last_active_layer(out + h)

        return out
