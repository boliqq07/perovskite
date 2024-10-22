# -*- coding: utf-8 -*-
from typing import Dict

import torch
from e3nn import o3

# Define a type alias
_SI = Dict[str, o3.Irreps]
_ST = Dict[str, torch.Tensor]
