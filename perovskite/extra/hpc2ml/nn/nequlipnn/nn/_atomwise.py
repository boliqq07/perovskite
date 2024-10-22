import logging
import warnings
from typing import Optional, List, Union

import torch
import torch.nn.functional
from e3nn.o3 import Irreps
from e3nn.o3 import Linear
from e3nn.util.jit import compile_mode
from torch_runstats.scatter import scatter, scatter_mean
from torch_scatter import scatter as sca

from hpc2ml.data.nequlip_support import AtomicDataDict
from hpc2ml.data.nequlip_support.AtomicDataDict import ST, SI
from hpc2ml.data.nequlip_support._transforms import TypeMapper
from ._graph_mixin import GraphModuleMixin

warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.jit"
                        )


class AtomwiseOperation(GraphModuleMixin, torch.nn.Module):
    def __init__(self, operation, out_field: str, irreps_in=None):
        super().__init__()
        # ## in_field = out_field
        self.operation = operation
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={out_field: operation.irreps_in},
            irreps_out={out_field: operation.irreps_out}, )

    def forward(self, data: ST) -> ST:
        data[self.out_field] = self.operation(data[self.out_field])
        return data


@compile_mode("script")
class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    in_field: str

    def __init__(
            self,
            in_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: Optional[str] = None,
            irreps_in: SI = None,
            irreps_out: Union[str, Irreps] = None,
    ):
        super().__init__()
        self.in_field = in_field
        out_field = out_field if out_field is not None else in_field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = {out_field: irreps_in[out_field]}
        else:
            irreps_out = {out_field: Irreps(irreps_out)}

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[in_field],
            irreps_out=irreps_out,
        )
        self.linear = Linear(
            irreps_in=self.irreps_in[in_field], irreps_out=self.irreps_out[out_field]
        )

    def forward(self, data: ST) -> ST:
        data[self.out_field] = self.linear(data[self.in_field])
        return data


@compile_mode("script")
class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float
    in_field: str
    out_field: str

    def __init__(
            self,
            in_field: str,
            out_field: Optional[str] = None,
            reduce: str = "sum",
            avg_num_atoms: float = 1.0,
            irreps_in: Optional[SI] = {},
    ):
        super().__init__()
        if irreps_in is None:
            irreps_in = {}

        assert reduce in ("sum", "mean", "normalized_sum")

        self.constant = 1.0

        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.in_field = in_field
        self.out_field = f"{reduce}_{in_field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.in_field]}
            if self.in_field in irreps_in else {},
        )

    def forward(self, data: ST) -> ST:
        data = AtomicDataDict.with_batch(data)

        if self.reduce == "mean":
            data[self.out_field] = scatter_mean(
                data[self.in_field], data[AtomicDataDict.BATCH_KEY], dim=0,
                dim_size=data[self.out_field].shape[0]
            )
        elif self.reduce == "sum":
            data[self.out_field] = scatter(
                data[self.in_field], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce,
                dim_size=data[self.out_field].shape[0]
            )
        else:
            data[self.out_field] = sca(
                data[self.in_field], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce,
                dim_size=data[self.out_field].shape[0]
            )

        if self.constant != 1.0:
            data[self.out_field] = data[self.out_field] * self.constant
        return data


@compile_mode("script")
class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Args:
        in_field: the per-atom in_field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output in_field; defaults to ``in_field``.
    """

    in_field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
            self,
            in_field: str,
            num_types: int,
            type_names: List[str],
            shifts: Optional[List[float]],
            scales: Optional[List[float]],
            arguments_in_dataset_units: bool,
            out_field: Optional[str] = None,
            scales_trainable: bool = False,
            shifts_trainable: bool = False,
            irreps_in=None,
    ):
        super().__init__()
        if irreps_in is None:
            irreps_in = {}
        self.num_types = num_types
        self.type_names = type_names
        self.in_field = in_field
        self.out_field = f"shifted_{in_field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.in_field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.in_field]},
        )

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types) * shifts
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types) * scales
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        self.arguments_in_dataset_units = arguments_in_dataset_units

    def forward(self, data: ST) -> ST:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        in_field = data[self.in_field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[species_idx].view(-1, 1) + in_field
        data[self.out_field] = in_field
        return data

    def update_for_rescale(self, rescale_module):
        if hasattr(rescale_module, "related_scale_keys"):
            if self.out_field not in rescale_module.related_scale_keys:
                return
        if self.arguments_in_dataset_units and rescale_module.has_scale:
            logging.debug(
                f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
                f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
            with torch.no_grad():
                if self.has_scales:
                    self.scales.div_(rescale_module.scale_by)
                if self.has_shifts:
                    self.shifts.div_(rescale_module.scale_by)
            logging.debug(
                f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
