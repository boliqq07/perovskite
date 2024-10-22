# -*- coding: utf-8 -*-

from typing import Tuple, Dict, List, Union, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.core import Structure, Lattice, Molecule
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.in_memory_dataset import nested_iter
from torch_geometric.data.separate import separate

from hpc2ml.data.io import aaa
from hpc2ml.data.io.main import sparse_source_data
from hpc2ml.data.structuretodata import StructureToData


class MtBatchData(Batch):
    """
    One simple Batch data (torch_geometrics) integrate with generation and statistics,
    which is one higher wrapper of StructureToData.

    Examples:
        >>> mtdata = MtBatchData.from_data_list(batch)
        >>> data_list = MtBatchData.to_data_list()

    Examples:
        >>> mtdata = MtBatchData.from_batch(batch)
        >>> batch = MtBatchData.to_batch()

    Examples:
        >>> mtdata = MtBatchData.from_data(data,slices)
        >>> data,slices = MtBatchData.to_data()

    Examples:
        >>> mtdata = MtBatchData.from_atoms(atoms)
        >>> atoms = MtBatchData.to_atoms()

    Examples:
        >>> mtdata = MtBatchData.from_structure(structure)
        >>> structure = MtBatchData.to_structure()

    Examples:
        >>> mtdata = MtBatchData.from_sparse_source_data(...)
    See Also: StructureToData.

    Examples:
        >>> mtdata = MtBatchData.from_local_data_dict(file1, convert=StructureToData(),
        ... msg_name=(), save=False, store_root_dir=".",)

    """

    @classmethod
    def from_batch(cls, batch_data: Batch) -> "MtBatchData":
        """From torch_geometrics's Batch."""
        res = [batch_data.get_example(i) for i in range(batch_data.num_graphs)]
        return cls.from_data_list(res)

    def to_batch(self) -> Batch:
        """To torch_geometrics's Batch."""
        res = [self.get_example(i) for i in range(self.num_graphs)]
        return Batch.from_data_list(res)

    @staticmethod
    def _len(slices) -> int:
        if slices is None:
            return 1
        for _, value in nested_iter(slices):
            return len(value) - 1
        return 0

    @classmethod
    def from_data(cls, data: Data, slices) -> "MtBatchData":
        """From torch_geometrics's Data and slices (dict)."""
        l = cls._len(slices)
        res = [separate(
            cls=data.__class__,
            batch=data,
            idx=idx,
            slice_dict=slices,
            decrement=False, ) for idx in range(l)]
        return cls.from_data_list(res)

    def to_data(self) -> Tuple[BaseData, Mapping]:
        """To torch_geometrics's Data and slices (dict)."""
        res = [self.get_example(i) for i in range(self.num_graphs)]

        data, slices, _ = collate(
            self[0].__class__,
            data_list=res,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def add_prop(self, key, value, slice_type="atom"):
        """Add properties could be sliced."""
        if slice_type in ["atom", "node"]:
            ci = self._inc_dict["x"]
            si = self._slice_dict["x"]
        elif slice_type == "sample":
            ci = self._inc_dict["natoms"]
            si = self._slice_dict["natoms"]
        elif slice_type == "edge":
            ci = self._inc_dict["edge_index"]
            si = self._slice_dict['edge_index']
        else:
            raise NotImplementedError

        setattr(self, key, value)
        self.keys.append(key)
        self._slice_dict.update({key:si})
        self._inc_dict.update({key:ci})

    @classmethod
    def from_atoms(cls, atoms: Union[Atoms, List[Atoms]], convert=StructureToData(),
                   msg_name=None, remain_old_edge=True, **kwargs, ) -> "MtBatchData":
        """From ase atoms."""
        if msg_name is None:
            msg_name = ("energy", "energies", "forces", "stress")

        if isinstance(atoms, Atoms):
            atoms = [atoms, ]
        else:
            atoms = atoms

        convert.just_predefined = False

        add_kw = {name: [i.calc.results[name] if i.calc is not None and name in i.calc.results else None for i in atoms]
                  for name in msg_name}

        if remain_old_edge:
            msg_name2 = ("edge_index",)
            add_kw2 = {name: [i.info[name] if name in i.info else None for i in atoms] for name in msg_name2}
            add_kw.update(add_kw2)

        add_kw.update(kwargs)
        kwargs = {k: v for k, v in add_kw.items() if v[0] is not None}

        st = [aaa.get_structure(i) for i in atoms]
        return cls.from_structure(structure=st, convert=convert, **kwargs, )

    def to_atoms(self, extra_fields=None, ) -> Union[List[Atoms], Atoms]:
        """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

        For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
        an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
        exist in self, a single ``ase.Atoms`` object is created.

        Args:
            extra_fields: fields other than those handled explicitly (currently
                those defining the structure as well as energy, per-atom energy,
                and forces) to include in the output object. Per-atom (per-node)
                quantities will be included in ``arrays``; per-graph and per-edge
                quantities will be included in ``info``.

        Returns:
            A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
            and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
        """
        if extra_fields is None:
            # extra_fields = []
            extra_fields = ["edge_index", ]

        positions = getattr(self, "pos")

        if positions.device != torch.device("cpu"):
            raise TypeError(
                "Explicitly move this `AtomicData` to CPU using `.to()` before calling `to_ase()`.")

        batch = getattr(self, "batch")
        n_batches = batch.max() + 1
        batch_atoms = []
        for batch_idx in range(n_batches):

            data = self.get_example(batch_idx)
            pbc = getattr(data, "pbc", None)
            atomic_nums = getattr(data, "z", None)
            pos = getattr(data, "pos", None)
            cell = getattr(data, "cell", None)
            edge_index = getattr(data, "edge_index", None)

            energy = getattr(data, "energy", None)
            energies = getattr(data, "energies", None)
            forces = getattr(data, "forces", None)
            stress = getattr(data, "stress", None)

            mol = Atoms(
                numbers=atomic_nums.cpu().numpy().ravel(),  # must be flat for ASE
                positions=pos.cpu().numpy(),
                cell=torch.squeeze(cell).cpu().numpy() if cell is not None else None,
                pbc=pbc.view(-1).cpu().numpy() if pbc is not None else None, )

            # add other information
            fields = {}
            if energies is not None:
                fields["energies"] = energies.cpu().numpy()
            if energy is not None:
                fields["energy"] = energy.cpu().numpy()
            if forces is not None:
                fields["forces"] = forces.cpu().numpy()
            if stress is not None:
                stress = stress.cpu().numpy()
                if self["stress"].shape[1] == 3:
                    fields["stress"] = full_3x3_to_voigt_6_stress(stress)
                else:
                    fields["stress"] = stress

            mol.calc = SinglePointCalculator(mol, **fields)

            # add other information
            for key in extra_fields:
                if key == "edge_index":
                    mol.info["edge_index"] = edge_index.cpu().numpy().astype(np.long)
                else:
                    raise RuntimeError(
                        f"Extra field `{key}` isn't registered as node/edge/graph"
                    )

            batch_atoms.append(mol)

        if batch is not None:
            return batch_atoms
        else:
            assert len(batch_atoms) == 1
            return batch_atoms[0]

    @classmethod
    def from_sparse_source_data(cls, source_file="vasprun.xml", source_path="./pure_opt",
                                fmt="vasprun_traj", convert=StructureToData(),
                                msg_name=(), store=False, store_root_dir=".", **kwargs) -> "MtBatchData":
        """
        Sparse data by different function.

        Args:
            store: bool, save.
            store_root_dir:str, store path.
            source_file: file name to sparse.
            source_path: (str,list), path or paths.
            fmt: str, load function named  "sparse_{fmt}" in from ``hpc2ml.data.io`` .
            n_jobs: int, the parallel number to load,
            msg_name: tuple, names to convert.
            convert: StructureToData, converter.
            **kwargs: dict, the parameter in "sparse_{fmt}".

        Returns:
            data_dict:dict, data
        """
        dicts = sparse_source_data(source_file=source_file, source_path=source_path, fmt=fmt,
                                   **kwargs)
        convert.data_dict = dicts
        if store:
            res = convert.transform_data_dict_and_save(msg_name=msg_name, store_root_dir=store_root_dir, )
        else:
            res = convert.transform_data_dict(msg_name=msg_name)
        return cls.from_data_list(res)

    @classmethod
    def from_local_data_dict(cls, file: Union[Dict, str], convert=StructureToData(),
                             msg_name=(), save=False, store_root_dir=".", ) -> "MtBatchData":
        """From dict data dumped by pandas, in local disk"""
        if isinstance(file, str):
            dicts = pd.read_pickle(file)
        else:
            dicts = file

        convert.data_dict = dicts
        if save:
            res = convert.transform_data_dict_and_save(msg_name=msg_name, store_root_dir=store_root_dir, )
        else:
            res = convert.transform_data_dict(msg_name=msg_name)
        return cls.from_data_list(res)

    @classmethod
    def from_structure(cls, structure: Union[Structure, List[Structure]], convert=StructureToData(),
                       **kwargs, ) -> "MtBatchData":
        """From pymatgen Structure"""
        if isinstance(structure, Structure):
            structure = [structure, ]
        convert.just_predefined = False
        res = convert.transform(structure=structure, **kwargs)
        return cls.from_data_list(res)

    def to_structure(self, extra_fields=None, ) -> Union[Tuple[List[Structure], Dict], Tuple[Structure, Dict]]:
        """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

        For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
        an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
        exist in self, a single ``ase.Atoms`` object is created.

        Args:
            extra_fields: fields other than those handled explicitly (currently
                those defining the structure as well as energy, per-atom energy,
                and forces) to include in the output object. Per-atom (per-node)
                quantities will be included in ``arrays``; per-graph and per-edge
                quantities will be included in ``info``.

        Returns:
            A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
            and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
        """
        if extra_fields is None:
            extra_fields = ["edge_index", ]

        positions = getattr(self, "pos")

        if positions.device != torch.device("cpu"):
            raise TypeError(
                "Explicitly move this `AtomicData` to CPU using `.to()` before calling `to_ase()`.")

        batch_atoms = []
        add_msg = []
        names = set()

        batch = getattr(self, "batch")
        n_batches = batch.max() + 1
        for batch_idx in range(n_batches):
            data = self.get_example(batch_idx)
            pbc = getattr(data, "pbc", None)
            atomic_nums = getattr(data, "z", None)
            pos = getattr(data, "pos", None)
            cell = getattr(data, "cell", None)
            edge_index = getattr(data, "edge_index", None)

            energy = getattr(data, "energy", None)
            energies = getattr(data, "energies", None)
            forces = getattr(data, "forces", None)
            stress = getattr(data, "stress", None)

            if cell is not None:

                lattice = Lattice(matrix=torch.squeeze(cell).cpu().numpy(),
                                  pbc=pbc.view(-1).cpu().numpy())
                mol = Structure(
                    lattice=lattice,
                    species=atomic_nums.cpu().numpy().ravel(),
                    coords=pos.cpu().numpy(),
                    coords_are_cartesian=True)
            else:
                mol = Molecule(
                    species=atomic_nums.cpu().numpy().ravel(),
                    coords=pos.cpu().numpy(), )

            fields = {}
            if energies is not None:
                fields["energies"] = energies.cpu().numpy()
            if energy is not None:
                fields["energy"] = energy.cpu().numpy()
            if forces is not None:
                fields["forces"] = forces.cpu().numpy()
            if stress is not None:
                stress = stress.cpu().numpy()
                if self["stress"].shape[1] == 3:
                    fields["stress"] = full_3x3_to_voigt_6_stress(stress)
                else:
                    fields["stress"] = stress

            # add other information
            for key in extra_fields:
                if key == "edge_index":
                    fields[key] = edge_index.cpu().numpy().astype(np.long)
                else:
                    raise RuntimeError(
                        f"Extra field `{key}` isn't registered as node/edge/graph"
                    )

            batch_atoms.append(mol)
            add_msg.append(fields)
            names = names | set(fields.keys())

        names = list(names)

        kw = {i: [mi[i] for mi in add_msg] for i in names}

        if batch is not None:
            return batch_atoms, kw
        else:
            assert len(batch_atoms) == 1
            return batch_atoms[0], kw

    def to_atomicdatadict(self: Data, exclude_keys=tuple()) -> dict:
        """To dict to compare with AtomicDataDict"""
        keys = self.keys

        return {
            k: self[k]
            for k in keys
            if (k not in exclude_keys and self[k] is not None and isinstance(self[k], torch.Tensor))
        }

    def _per_static(self, fi, ana_mode, arr, inverse=False):
        if "per_atom" in ana_mode:
            natoms = getattr(self, "natoms")
            assert len(arr) == len(natoms), \
                f"Just features with same size of graph (sample number) could be divided.\n" \
                f"{fi} can't use 'per_atom' prefix."
            if inverse:
                arr = arr * natoms
            else:
                arr = arr / natoms
            res = {"per_atom": True}
        else:
            res = {"per_atom": False}
        return arr, res

    def statistics(
            self,
            fields: Union[str, List[str]],
            modes: Union[str, List[str]],
            unbiased: bool = True,
            by_dim=False,
    ) -> Dict[str, Dict[str, Union[float, torch.Tensor]]]:
        """
        statics the data.

        Args:
            fields: (str,list of str), names of property.
            modes: (str,list of str), names of in ["rms","unique","count","mean_std","max_min","self",
            "per_atom","per_atom",].
            unbiased: bool, unbiased for std.
            by_dim: (bool,list of bool) . statics according dim 0.

        Returns:
            data:dict
        """

        out = {}

        if not isinstance(fields, (list, tuple)):
            fields = [fields, ]

        if not isinstance(modes, (list, tuple)):
            modes = [modes, ]

        if not isinstance(by_dim, (list, tuple)):
            by_dim = [by_dim, ]

        if len(fields) == 1 and len(modes) > 1:
            fields = [fields[0]] * len(modes)
        elif len(fields) > 1 and len(modes) == 1:
            modes = [modes[0]] * len(fields)

        elif len(by_dim) == 1:
            by_dim = [by_dim[0]] * len(fields)

        assert len(fields) == len(modes) == len(by_dim)

        def up_add(fi, name, value):
            if fi in out:
                out[fi].update({name: value})
            else:
                out[fi] = {name: value}

        for fi, ana_mode, by_di in zip(fields, modes, by_dim):
            if not hasattr(self, fi):
                raise KeyError(f"No key named {fi}.")
            else:
                arr = getattr(self, fi)

                arr, perm = self._per_static(fi, ana_mode, arr, inverse=False)
                for k, v in perm.items():
                    up_add(fi, k, v)

                # compute statistics
                if "unique" in ana_mode:
                    # count integers
                    _, counts = torch.unique(torch.flatten(arr), return_counts=True, sorted=True)
                    up_add(fi, "unique", counts)

                if "count" in ana_mode:
                    # count integers
                    counts = arr.shape[0]
                    up_add(fi, "count", counts)

                elif "rms" in ana_mode:
                    # root-mean-square
                    if by_di:
                        counts = torch.sqrt(torch.mean(arr * arr, dim=0)).detach()
                    else:
                        counts = float(torch.sqrt(torch.mean(arr * arr)))
                    up_add(fi, "rms", counts)

                elif "mean_std" in ana_mode:
                    # mean and std
                    if by_di:
                        mean = torch.mean(arr, dim=0).detach()
                        std = torch.std(arr, dim=0, unbiased=unbiased).detach()
                    else:
                        mean = float(torch.mean(arr))
                        std = float(torch.std(arr, unbiased=unbiased))
                    up_add(fi, "mean", mean)
                    up_add(fi, "std", std)

                elif "max_min" in ana_mode:
                    if by_di:
                        maxs = torch.max(arr, dim=0).detach()
                        mins = torch.min(arr, dim=0).detach()
                    else:
                        maxs = float(torch.max(arr))
                        mins = float(torch.min(arr))
                    up_add(fi, "max", maxs)
                    up_add(fi, "min", mins)

                elif ana_mode is None or "self" in ana_mode:
                    up_add(fi, "self", arr)
                else:
                    raise NotImplementedError(f"Cannot handle statistics mode {ana_mode}")

        return out

    def scale(self, dct: Dict[str, Dict[str, Union[float, torch.Tensor]]]):
        """Scale the properties by the output of statistics."""
        for fi, ana_mode_dict in dct.items():
            if not hasattr(self, fi):
                pass
            else:
                arr = getattr(self, fi)

                ana_mode_name = "".join([i for i in ana_mode_dict.keys() if ana_mode_dict[i] is not False ])
                arr, _ = self._per_static(fi, ana_mode_name, arr, inverse=False)

                if "mean" in ana_mode_dict and "std" in ana_mode_dict:
                    arr = (arr - ana_mode_dict["mean"]) / ana_mode_dict["std"]
                elif "max" in ana_mode_dict and "min" in ana_mode_dict:
                    sc = ana_mode_dict["max"] - ana_mode_dict["min"]
                    arr = (arr - ana_mode_dict["min"]) / sc
                else:
                    pass
                setattr(self, fi, arr)
        return self

    def unscale(self, dct: Dict[str, Dict[str, Union[float, torch.Tensor]]]):
        """Un-scale the properties by the output of statistics."""
        for fi, ana_mode_dict in dct.items():
            if not hasattr(self, fi):
                pass
            else:
                arr = getattr(self, fi)

                if "mean" in ana_mode_dict and "std" in ana_mode_dict:
                    arr = (arr * ana_mode_dict["std"]) + ana_mode_dict["mean"]
                elif "max" in ana_mode_dict and "min" in ana_mode_dict:
                    sc = ana_mode_dict["max"] - ana_mode_dict["min"]
                    arr = arr * sc + ana_mode_dict["min"]
                else:
                    pass
                ana_mode_name = "".join([i for i in ana_mode_dict.keys() if ana_mode_dict[i] is not False ])
                arr, _ = self._per_static(fi, ana_mode_name, arr, inverse=True)

                setattr(self, fi, arr)

        return self

    def filter_index(self, prop: str, condition: Callable, independence: bool):
        """Filter by condition and return index."""
        batch = getattr(self, "batch")
        if independence:
            index = torch.Tensor([condition(getattr(i, prop)) for i in range(batch.max() + 1)])
        else:
            index = torch.Tensor(condition(getattr(self, prop)))
            assert len(index) == self.num_graphs, "only index with size == num_graphs could be used."
        return index

    def filter(self, prop: str, condition: Callable, independence: bool):
        """Filter by condition and return new."""
        index = self.filter_index(prop=prop, condition=condition, independence=independence)
        return self[index]

    def plot_show(self, fields: Union[str, List[str]] = None,
                  modes: Union[str, List[str]] = None,
                  by_dim: Union[bool, List[bool]] = None,
                  unbiased: bool = False, ):
        """plot

        Args:
            fields: (str,list of str), names of property.
            modes: (str,list of str), names of in ["rms","unique","count","mean_std","max_min","self",
            "per_atom","per_atom",].
            unbiased: bool, unbiased for std.
            by_dim: (bool,list of bool) . statics according dim 0.
        """
        if by_dim is None:
            by_dim = [False, True, False]
        if fields is None:
            fields = ["energy", "forces", "stress"]
        if modes is None:
            modes = ["self", "mean_std", "max_min"]

        import numpy as np

        import matplotlib.pyplot as plt

        dct = self.statistics(fields=fields, modes=modes, unbiased=unbiased, by_dim=by_dim)
        sp = {}
        sp1 = {}
        sp2 = {}
        for k, v in dct.items():
            for ki, vi in v.items():
                if isinstance(vi, (int, float)) or (isinstance(vi, torch.Tensor) and vi.ndim == 0):
                    sp.update({f"{k} ({ki})": np.array(vi)})
                elif isinstance(vi, torch.Tensor) and vi.ndim == 1:
                    sp1.update({f"{k} ({ki})": vi.view(-1).numpy()})
                elif isinstance(vi, torch.Tensor) and vi.ndim == 2:
                    sp2.update({f"{k} ({ki})": vi.numpy()})

                else:
                    print("Drop" f"{k}-{ki} due to large dim of tensor or error type.")
        start = 1 if len(sp) > 0 else 0
        shape = start + len(sp1) + len(sp2)

        plt.figure(figsize=(max((2.4 * len(sp1), 4.8)), 2 * shape))
        plt.subplot(shape, 1, 1, )
        plt.subplots_adjust(hspace=1)
        y = np.array(list(sp.values()))

        p1 = plt.bar(np.array(list(sp.keys())), y)
        plt.bar_label(p1, labels=y, fmt="%.2f", label_type='edge', padding=0, )
        plt.title("single properties")

        p = start
        for k, v in sp1.items():
            p = p + 1
            x_ = np.arange(len(v))
            plt.subplot(shape, 1, p)
            plt.xticks(x_, x_)
            plt.scatter(x_, v, marker="o" )
            plt.title(k)

        for k, v in sp2.items():
            p = p + 1
            plt.subplot(shape, 1, p)
            plt.imshow(v)
            plt.title(k)
        plt.savefig("data_range.jpg")
        try:
            plt.show()
        except BaseException:
            pass

    def plot_show_simple(self, fields: Union[str, List[str]] = "energy",
                         modes: Union[str, List[str]] = "self",
                         by_dim=False,
                         unbiased: bool = False, ):
        """quick plot energy.
        """
        self.plot_show(fields=fields, modes=modes, unbiased=unbiased, by_dim=by_dim)


def to_atomicdatadict(data: [Batch, Data, MtBatchData], exclude_keys=tuple()) -> dict:
    """quick to dict."""
    keys = data.keys
    return {k: data[k]
            for k in keys
            if (k not in exclude_keys and data[k] is not None and isinstance(data[k], torch.Tensor))
            }


def filter_upper(arr: torch.Tensor, theshold: float):
    return arr > theshold


def filter_lower(arr: torch.Tensor, theshold: float):
    return arr < theshold


def filter_equal(arr: torch.Tensor, theshold: float):
    return arr == theshold
