import torch

from hpc2ml.data.nequlip_support.AtomicDataDict import ST
from ._graph_mixin import GraphModuleMixin


class SaveForOutput(torch.nn.Module, GraphModuleMixin):
    """Copy a in_field and disconnect it from the autograd graph.

    Copy a in_field and disconnect it from the autograd graph, storing it under another key for inspection as part of the models output.

    Args:
        in_field: the in_field to save
        out_field: the key to put the saved copy in
    """

    in_field: str
    out_field: str

    def __init__(self, in_field: str, out_field: str, irreps_in=None):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)
        self.irreps_out[out_field] = self.irreps_in[in_field]
        self.in_field = in_field
        self.out_field = out_field

    def forward(self, data: ST) -> ST:
        data[self.out_field] = data[self.in_field].detach().clone()
        return data
