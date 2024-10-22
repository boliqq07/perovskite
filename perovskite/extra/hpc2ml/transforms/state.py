# -*- coding: utf-8 -*-

# @Time  : 2022/5/20 13:41
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import torch
from torch_geometric.transforms import BaseTransform


class Add2state(BaseTransform):
    def __call__(self, data):
        data.state_attr = torch.tensor([0.0, 0.0]).reshape(1, -1)
        return data
