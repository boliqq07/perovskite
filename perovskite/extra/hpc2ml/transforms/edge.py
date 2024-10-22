import warnings
from typing import List, Optional, Dict

import ase.data as ase_data
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from hpc2ml.function.support import add_edge_no_pbc, add_edge_no_pbc_from_index, \
    add_edge_pbc_from_index, add_edge_pbc, \
    distribute_edge


class AddEdgeNoPBC(BaseTransform):
    """Add edge index, with no pbc."""

    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, data):
        if hasattr(data, "edge_index"):
            data = add_edge_no_pbc_from_index(data)
        else:
            data = add_edge_no_pbc(data, cutoff=self.cutoff)
        return data


class AddEdgePBC(BaseTransform):
    """Add edge index, with no pbc."""

    def __init__(self, cutoff=5.0, max_neighbors=16):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def __call__(self, data):
        if hasattr(data, "edge_index"):
            data = add_edge_pbc_from_index(data)
        else:
            data = add_edge_pbc(data, cutoff=self.cutoff, max_neighbors=self.max_neighbors)
        return data


class AddAttrToWeight(BaseTransform):
    """Add edge index, with no pbc."""

    def __call__(self, data):
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)
        return data


class AddAttrSumToAttrAndWeight(BaseTransform):
    """Add edge index, with no pbc."""

    def __call__(self, data):
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)
            data.edge_attr = data.edge_weight.reshape(-1, 1)
        return data


class AddAttrSumToAttr(BaseTransform):
    """Add edge index, with no pbc."""

    def __call__(self, data):
        assert hasattr(data, "edge_attr") and data.edge_attr is not None
        assert data.edge_attr.shape[1] == 3
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)

        wei = data.edge_weight

        data.edge_attr = torch.cat((wei.reshape(-1, 1), data.edge_attr), dim=1)

        return data


class DistributeEdgeAttr(BaseTransform):
    """Compact with rotnet network (deep potential)"""

    def __init__(self, r_cs=2.0, r_c=6.0, cat_weight_attr=True):
        super().__init__()
        self.r_cs = r_cs
        self.r_c = r_c
        self.cat_weight_attr = cat_weight_attr

    def __call__(self, data):
        data = distribute_edge(data, self.r_cs, self.r_c)
        return data


class AtomNumToType(BaseTransform):
    num_types: int
    chemical_symbol_to_type: Optional[Dict[str, int]]
    type_names: List[str]
    _min_Z: int
    from hpc2ml.data import AtomicDataDict

    def __init__(self, *args, type_names: Optional[List[str]] = None,
                 chemical_symbol_to_type: Optional[Dict[str, int]] = None,
                 chemical_symbols: Optional[List[str]] = None, **kwargs):
        super(AtomNumToType, self).__init__(*args, **kwargs)

        if chemical_symbol_to_type is None:
            if chemical_symbols is None:
                chemical_symbols = ase_data.chemical_symbols
            atomic_nums = [ase_data.atomic_numbers[sym] for sym in chemical_symbols]
            chemical_symbols = [e[1] for e in sorted(zip(atomic_nums, chemical_symbols))]
            chemical_symbol_to_type = {k: i for i, k in enumerate(chemical_symbols)}
            del chemical_symbols

        # Build from chem->type mapping, if provided
        self.chemical_symbol_to_type = chemical_symbol_to_type

        # Validate
        for sym, type in self.chemical_symbol_to_type.items():
            assert sym in ase_data.atomic_numbers, f"Invalid chemical symbol {sym}"
            assert 0 <= type, f"Invalid type number {type}"
        assert set(self.chemical_symbol_to_type.values()) == set(
            range(len(self.chemical_symbol_to_type))
        )
        if type_names is None:
            # Make type_names
            type_names = [None] * len(self.chemical_symbol_to_type)
            for sym, type in self.chemical_symbol_to_type.items():
                type_names[type] = sym
        else:
            # Make sure they agree on types
            # We already checked that chem->type is contiguous,
            # so enough to check length since type_names is a list
            assert len(type_names) == len(self.chemical_symbol_to_type)
        # Make mapper array
        valid_atomic_numbers = [
            ase_data.atomic_numbers[sym] for sym in self.chemical_symbol_to_type
        ]
        self._min_Z = min(valid_atomic_numbers)
        self._max_Z = max(valid_atomic_numbers)
        Z_to_index = torch.full(
            size=(1 + self._max_Z - self._min_Z,), fill_value=-1, dtype=torch.long
        )
        for sym, type in self.chemical_symbol_to_type.items():
            Z_to_index[ase_data.atomic_numbers[sym] - self._min_Z] = type
        self._Z_to_index = Z_to_index
        self._index_to_Z = torch.zeros(
            size=(len(self.chemical_symbol_to_type),), dtype=torch.long
        )
        for sym, type_idx in self.chemical_symbol_to_type.items():
            self._index_to_Z[type_idx] = ase_data.atomic_numbers[sym]
        self._valid_set = set(valid_atomic_numbers)

        # check
        if type_names is None:
            raise ValueError(
                "None of chemical_symbols, chemical_symbol_to_type, nor type_names was provided; "
                "exactly one is required")
        # validate type names
        assert all(n.isalnum() for n in type_names), "Type names must contain only alphanumeric characters"
        # Set to however many maps specified -- we already checked contiguous
        self.num_types = len(type_names)
        # Check type_names
        self.type_names = type_names

    def __call__(self, data: Data, types_required: bool = True) -> Data:
        if hasattr(data, self.AtomicDataDict.ATOM_TYPE_KEY):
            if hasattr(data, self.AtomicDataDict.ATOMIC_NUMBERS_KEY):
                warnings.warn(
                    "Data contained both ATOM_TYPE_KEY and ATOMIC_NUMBERS_KEY; ignoring ATOMIC_NUMBERS_KEY"
                )
        elif hasattr(data, self.AtomicDataDict.ATOMIC_NUMBERS_KEY):
            assert (
                    self.chemical_symbol_to_type is not None
            ), "Atomic numbers provided but there is no chemical_symbols/chemical_symbol_to_type mapping!"
            atomic_numbers = getattr(data, self.AtomicDataDict.ATOMIC_NUMBERS_KEY)

            delattr(data, self.AtomicDataDict.ATOMIC_NUMBERS_KEY)

            setattr(data, self.AtomicDataDict.ATOM_TYPE_KEY, self._transform(atomic_numbers))
        else:
            if types_required:
                raise KeyError(
                    "Data doesn't contain any atom type information (ATOM_TYPE_KEY or ATOMIC_NUMBERS_KEY)"
                )
        return data

    def _transform(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(atomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )

        return self._Z_to_index.to(device=atomic_numbers.device)[
            atomic_numbers - self._min_Z
            ]
