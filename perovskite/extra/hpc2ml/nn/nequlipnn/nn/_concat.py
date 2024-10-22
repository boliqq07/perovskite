from typing import List

import torch
from e3nn import o3

from hpc2ml.data.nequlip_support.AtomicDataDict import ST
from ._graph_mixin import GraphModuleMixin


class Concat(GraphModuleMixin, torch.nn.Module):
    """Concatenate multiple fields into one."""

    def __init__(self, in_fields: List[str], out_field: str, irreps_in=None):
        super().__init__()
        if irreps_in is None:
            irreps_in = {}
        self.in_fields = list(in_fields)
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=self.in_fields)
        self.irreps_out[self.out_field] = sum(
            (self.irreps_in[k] for k in self.in_fields), o3.Irreps()
        )

    def forward(self, data: ST) -> ST:
        data[self.out_field] = torch.cat([data[k] for k in self.in_fields], dim=-1)
        return data
