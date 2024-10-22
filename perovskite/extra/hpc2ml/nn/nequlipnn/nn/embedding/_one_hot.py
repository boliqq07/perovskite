import torch
import torch.nn.functional
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from hpc2ml.data.nequlip_support import AtomicDataDict
from hpc2ml.data.nequlip_support.AtomicDataDict import ST
from .._graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.
    out_field in [AtomicDataDict.NODE_ATTRS_KEY,AtomicDataDict.NODE_FEATURES_KEY,]

    Args:
        copy_features: If ``True`` (default), copy out_field to another.
    """

    num_types: int
    copy_features: bool
    out_field: str
    another_field: str

    def __init__(
            self,
            irreps_in=None,
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            num_types: int = 50,
            copy_features: bool = True,

    ):
        super().__init__()
        self.num_types = num_types
        self.copy_features = copy_features
        self.out_field = out_field

        another_field = [i for i in [AtomicDataDict.NODE_FEATURES_KEY,
                                     AtomicDataDict.NODE_ATTRS_KEY] if i != out_field][0]
        self.another_field = another_field

        # Output irreps are num_types even (invariant) scalars nx0e
        irreps_out = {self.out_field: Irreps([(self.num_types, (0, 1))])}

        if self.copy_features:
            irreps_out[self.another_field] = irreps_out[self.out_field]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: ST) -> ST:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[self.out_field] = one_hot
        if self.copy_features:
            data[self.another_field] = one_hot
        return data
