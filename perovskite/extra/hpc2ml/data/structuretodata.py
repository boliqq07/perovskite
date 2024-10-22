# -*- coding: utf-8 -*-

# @Time  : 2022/5/9 17:46
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

"""

Abstract classes for building graph representations consist with ``pytorch-geometric``.

All the Graph in this part should return data as following:

Each Graph data (for each structure):

``x``: Node feature matrix. np.ndarray, with shape [num_nodes, nfeat_node]

``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

``energy``: target. np.ndarray, shape (1, num_target) , default shape (1,)

#  ``y``: target 1. (alias edge_energy) np.ndarray, shape (1, num_target) , default shape (1,)

``z``: atom numbers (alias atomic_numbers). np.ndarray, with shape [num_nodes,]

``cell``: cell matrix. np.ndarray,  with shape [3, 3]

``state_attr``: state feature. np.ndarray, shape (1, nfeat_state)

``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long

                It is neighbor_index and center_index sequentially.

``edge_weight``: Edge feature matrix. np.ndarray,  with shape [num_edges,]

# ``distance``: (alias edge_weight) distance matrix. np.ndarray,  with shape [num_edges, 3]

``edge_attr``: Edge feature matrix. np.ndarray,  with shape [num_edges, nfeat_edge]

# ``distance_vec``: (alias state_attr) distance 3D matrix (x,y,z). np.ndarray,  with shape [num_edges, 3]

``cell_offsets``: offset matrix. np.ndarray, with shape [3, 3]

``forces``: forces matrix per atom. np.ndarray,  with shape [num_nodes, 3]

``stress``: stress matrix per atom. np.ndarray,  with shape [6,]

``natoms``: (int) number of atom.

# ``tags``: tags,  not used.

# ``fixed``: fixed atoms, tags , not used.

"""
import os
import traceback
import warnings
from shutil import rmtree
from typing import List, Iterable, Union, Tuple, Sequence, Optional, Dict

import ase.data as ase_data
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure, Element
from pymatgen.optimization.neighbors import find_points_in_spheres  # this is ok
from torch_geometric.data import Data

from hpc2ml.data.io.main import sparse_source_data
from hpc2ml.data.preprocessing.pack import unpack, pack
from mgetool.tool import parallelize, batch_parallelize


def data_merge(data_sub: List[Data], merge_key=None):
    if merge_key is None:
        merge_key = ["state_attr", "edge_attr", "x"]

    data_base = data_sub[0]
    if len(data_sub) == 1:
        pass
    else:
        for di in data_sub[1:]:
            ks = di.keys
            for k in ks:
                v = getattr(di, k)
                if v is not None:
                    if k not in merge_key:
                        data_base.__setattr__(k, v)
                    else:
                        if not hasattr(data_base, k):
                            data_base.__setattr__(k, v)
                        elif isinstance(getattr(data_base, k), torch.Tensor) and isinstance(v, torch.Tensor):
                            v_base = getattr(data_base, k)
                            v_base_s = len(v_base.shape)
                            v_s = len(v.shape)
                            if v_base_s == 0 or v_s == 0 or all((v_base_s == 1, v_s == 1)):
                                data_base.__setattr__(k, torch.cat([v_base.flatten(), v.flatten()]))
                            elif all((v_base_s >= 2, v_s >= 2)) and v_base.shape[0] == v_base.shape[0]:
                                data_base.__setattr__(k, torch.cat([v_base, v], dim=1))
                            else:
                                data_base.__setattr__(k, v)
                        else:
                            data_base.__setattr__(k, v)
    return data_base


def data_dtype(data: Data) -> Data:
    ks = data.keys
    for k in ks:
        v = getattr(data, k)
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float64:
                data.__setattr__(k, v.float())
    return data


class StructureToData:
    """
    Converter to convert Structure (pymatgen) To Data (torch_geometrics).

    1. Use
        1.1 Use as normal. Where ``transform`` is for batch of data, and ``convert`` is for one single data.
        ``transform_and_save`` is for batch of data and save to local file (Match Dataset in torch_geometrics)

        Examples:
            >>> cvt = StructureToData()
            >>> res = cvt.transform(structure=structure_list,energy = energy_list)
            >>> res = cvt.transform_and_save(structure=structure_list,energy = energy_list)
            >>> resi = cvt.convert(structure=one_structure,energy=0.3)

            For batch of data, there could be some error data, self.support_ marks the correct/error data.
            >>> support_ = cvt.support_

        1.2 Use subclass directly.
        Where contain_base=True means use the base _convert in StructureToData to (x,z,pos,cell .etc)
        execpt the fraccoords message, contain_base=Fasle means just convert fraccoords meaasge.

        Examples:
            >>> cvt = PAddFracCoords(contain_base=True)
            >>> res = cvt.transform(structure=structure_list,energy = energy_list)


        1.3 Use with more subclass. Where The PAddStress add stress property form ext input,
         and PAddFracCoords add fraccoords from structure.

        Examples:
            >>> cvt = StructureToData(sub_converters=[PAddStress(),PAddFracCoords(),PAddPBCEdgeDistance()])
            >>> res = cvt.transform(structure=structure_list,energy = energy_list, stress = stress_list)

        Notes:
            It doesn't accept 3 or more deep!!!
            such as: PAddForce(PAddForce(PAddPBCEdgeSole())) is lost PAddPBCEdgeSole().

        1.4 Add subclass for use. (use the same as 1.3).
        This would generate one new calss. Where the StructureToData exit or any contain_base=True,
        the new class could contain_base.

        Examples:
            >>> cvt = StructureToData()+PAddFracCoords()+PAddPBCEdgeSole()+PAddForce()
            >>> cvt2 = PAddFracCoords(contain_base=True)+PAddFracCoords()
            >>> res = cvt.transform(structure=structure_list,energy = energy_list, stress = stress_list, forces=force_list)
            >>> res2 = cvt.transform(structure=structure_list,energy = energy_list, stress = stress_list, forces=force_list)

        Notes: The 1.3 and 1.4 are not compacted, in other wards, we don't accept 3 or more deep!!!
            such as: PAddForce()+PAddForce(PAddPBCEdgeSole()) is lost PAddPBCEdgeSole().

    2. Sparse source data.
        In normal, you can offer the raw data as the front 1 part. we suggest using the builltin sparse function
        to add offer data.

        Examples:
            >>> addsap = StructureToData(sub_converters=[PAddFracCoords(), PAddForce(),
            ...                                          PAddPBCEdgeDistance(cutoff=5.0),
            ...                                          PAddStress()])
            >>> from mgetool.imports.leaf import find_leaf_path
            >>> source_path = find_leaf_path("./data/") # paths list
            >>> res = addsap.sparse_source_data(source_file="vasprun.xml", source_path=source_path, fmt="vasprun_traj",
            ...                          store_path_file="temp_sparse_dict.pkl_pd", n_jobs=4)

            The data are stored in self.data_dict automatrically.
            If the data are large, you could store sparseed data and load it in other code or concel the front code.

            >>> res = addsap.read_sparse_dict("./data/st_and_ave_energy_couple.pkl_pd") # other code

            Then you could convert the data.

            Examples:
                >>> res0 = addsap.convert_data_dict(index=0) # convert the 0 sample in addsap.data_dict
                >>> res = addsap.transform_data_dict() # transform addsap.data_dict
                >>> addsap.transform_data_dict_and_save() # transform addsap.data_dict and store.


    2. function Extension
        All subclass can be used to add functionality.
        Except use the pre-defind subclass, you could define your extension,just need defined ``_convert`` function.

        Examples:
            >>> class PAddUserName(StructureToData):
            >>>     def __init__(self, *args, **kwargs):
            >>>         super().__init__(*args, **kwargs)
            >>>
            >>>     def _convert(self, data: Structure, **kwargs) -> Data:
            >>>         new_prop = ...
            >>>         return Data(new_prop_name=new_prop)


    """

    def __init__(self, sub_converters: Union["StructureToData", List["StructureToData"]] = None, merge_key=None,
                 n_jobs: int = 1,
                 batch_calculate: bool = True, just_predefined=True, contain_base=False,
                 batch_size: int = 1000, tq=True,):
        """

        Args:
            sub_converters: list, list of converter.
            merge_key: list of str, name of msg should be merged if different convert.
            n_jobs: int, jobs (we suggest n_jobs=1)
            batch_calculate: int, batch calculate or not.
            just_predefined: bool, if false, try to convert all msg, even if you don't define the conversion method.
            batch_size:int, batch_size.
        """

        self.n_jobs = n_jobs
        self.batch_calculate = batch_calculate
        self.batch_size = batch_size
        self.merge_key = merge_key
        self.data_dict = {}
        self.source_support = None
        self.just_predefined = just_predefined
        self.contain_base = contain_base
        self.tq = tq

        if self.__class__.__name__ == "StructureToData":
            assert contain_base == False, "StructureToData is the base and can't contain_base."

        if isinstance(sub_converters, StructureToData):
            sub_converters = [sub_converters, ]

        if contain_base and sub_converters is not None:
            self.sub_converters = [StructureToData()] + sub_converters
        else:
            if contain_base and sub_converters is None:
                self.sub_converters = [StructureToData()]

        self.sub_converters = sub_converters

        if self.sub_converters is not None:
            for i in self.sub_converters:
                assert i.__class__.__name__ != self.__class__.__name__, f"Not accept itself as sub_converter."
                if i.sub_converters is not None:
                    warnings.warn(f"We don't accept 3 or more deep!!!, {i.sub_converters} is lost, change you code!!!",
                                  UserWarning)
                i.sub_converters = None  # remove deeper converter.
                i.n_jobs = 1

    def __str__(self):
        sname = str([i.__class__.__name__ for i in self.sub_converters])
        return f"{self.__class__.__name__}({sname})"

    def __add__(self, *args):
        """Return one joined converter named CatStructureToData, and args are set into sub_converters.


        1. If the converters contains StructureToData or contain_base is true in any converters,
        the returned CatStructureToData would contain base _convert else not.
        2. Remove StructureToData in args.
        2.

        """
        assert len(args) >= 1, "add accept 1 or more input."

        args = [self, ] + list(args)

        _convert = lambda *args, **kwargs: None

        CatStructureToData = type("CatStructureToData", (StructureToData,), {"_convert": _convert})

        n_jobs = max([i.n_jobs for i in args])
        just_predefined = any([i.just_predefined for i in args])

        cb1 = [i.__class__.__name__ == "StructureToData" for i in args]
        cb2 = [i.contain_base for i in args]
        cb = cb1 + cb2
        cb = any(cb)

        merge_key = []
        [merge_key.extend(i.merge_key) for i in args if i.merge_key is not None]
        merge_key = merge_key if len(merge_key) > 0 else None

        args2 = []
        for i in args:
            if i.__class__.__name__ == "CatStructureToData":
                args2.extend(i.sub_converters)
            else:
                args2.append(i)

        args2 = [i for i in args2 if i.__class__.__name__ != "StructureToData"]  # remove StructureToData

        for i in args2:
            if i.sub_converters is not None:
                warnings.warn(f"We don't accept 3 or more deep!!!, {i.sub_converters} is lost, change you code!!!",
                              UserWarning)
            i.sub_converters = None  # remove deeper converter.
            i.n_jobs = 1
            i.contain_base = False

        cstd = CatStructureToData(sub_converters=args2, merge_key=merge_key, n_jobs=n_jobs,
                                  just_predefined=just_predefined, contain_base=cb)

        return cstd

    def __call__(self, *args, **kwargs):
        return self.convert(*args, **kwargs)

    def sparse_source_data(self, source_file: str = "vasprun.xml", source_path: Union[Sequence, str] = ".",
                           fmt: str = "vasprun", store_path_file: str = "temp_sparse_dict.pkl_pd", n_jobs=None,
                           mark_label="", **kwargs):
        """
        Sparse source data file by different function and store msg to self.data_dict temporary,
        and store to store_path_file (if needed).

        Args:
            store_path_file: str, store filename.
            mark_label: str, mark to distinguish different data batch or source.
            source_file: file name to sparse.
            source_path: (str,list), path or paths.
            fmt: str, load function named  "sparse_{fmt}" in from ``hpc2ml.data.io`` .
            n_jobs: int, the parallel number to load,
            **kwargs: dict, the parameter in "sparse_{fmt}".

        Returns:
            data_dict:dict, data
        """
        if n_jobs is None:
            n_jobs = self.n_jobs
        dict_res = sparse_source_data(source_file=source_file, source_path=source_path, fmt=fmt, n_jobs=n_jobs,
                                      **kwargs)
        if mark_label != "":
            dict_res = {f"{mark_label}_{k}": v for k, v in dict_res.items()}

        self.data_dict = dict_res
        if isinstance(store_path_file, str):
            pd.to_pickle(dict_res, store_path_file)
        return dict_res

    def read_sparse_dict(self, path_file="temp_sparse_dict.pkl_pd"):
        """Read sparse data for local."""
        self.data_dict = pd.read_pickle(path_file)
        return self.data_dict

    def convert_data_dict(self, index=None, msg_name=None):
        """Convert one data in data_dict.
        default msg_name = ("structure", "energy", "state_attr", "forces","stress" , ...)"""

        if index is None:
            assert len(self.data_dict) == 1, "convert_data_dict is just used for 1 samples when index=None." \
                                             "Please denote which one to convert in data_dict by index."

        dati = self.data_dict[index]

        # if "y" in dati:
        #     dati["energy"] = dati.pop("y")

        if len(msg_name) > 0:
            new_dat = {}
            for mi in dati.keys():
                if mi in msg_name:
                    new_dat.update({mi: dati[mi]})
        else:
            new_dat = dati
        return self.convert(**new_dat)

    def transform_data_dict(self, msg_name=None):
        """Transform all data_dict.
        default msg_name = ("structure", "energy", "state_attr", "forces","stress" , ...)"""
        if msg_name is None:
            msg_name = ("structure", "energy", "y", "state_attr", "forces", "stress")

        ks, vs, k2s = self.upack()

        kv = {}

        if len(msg_name) > 0:
            for k, v in zip(ks, vs):
                if k in msg_name:
                    # k = "energy" if k == "y" else k
                    kv.update({k: v})
        else:
            for k, v in zip(ks, vs):
                # k = "energy" if k == "y" else k
                kv.update({k: v})

        if len(kv) == 0:
            raise ValueError("Not data to convert in self.data_dict")
        res = self.transform(**kv, note=k2s)
        return res

    def convert(self, structure: Structure, energy=None, state_attr=None, **kwargs) -> Data:
        """
        Convert single data.

        Args:
            structure: (Structure)
            energy: (float,int)
            state_attr: (float,int,np.ndarray)
            **kwargs: kwargs.

        Returns:
            data:(Data),pyg Data

        """
        if self.sub_converters is None:
            data = self._convert(structure, energy=energy, state_attr=state_attr, **kwargs)
        else:
            data_sub = [ci.convert(structure, energy=energy, state_attr=state_attr, **kwargs) for ci in
                        self.sub_converters]

            main_data = self._convert(structure, energy=energy, state_attr=state_attr, **kwargs)

            data_sub.append(main_data)

            data_sub = [i for i in data_sub if i is not None]

            if len(data_sub) == 0:
                data = None
            else:
                data = data_merge(data_sub, self.merge_key)
                data = data_dtype(data)

        return data

    def _wrapper(self, *args, **kwargs):
        """Deal with error data."""
        if "note" in kwargs:
            note = kwargs.pop("note")
        else:
            note = ""
        try:
            con = self.convert(*args, **kwargs)

            if isinstance(con, (List, Tuple)):
                if len(con) == 2 and isinstance(con[1], bool):
                    pass
                else:
                    con = (con, True)
            else:
                con = (con, True)
            return con

        except BaseException as e:
            print(f"Bad conversion for:{args[0].formula},note:{note}")
            traceback.print_exc()
            warnings.warn("Check the self.support_ to get the index of dropped error data.", UserWarning)
            return None, False

    def upack(self, data_dict=None):
        if data_dict is None:
            data_dict = self.data_dict
        ks, vs, k2s = unpack(data_dict)
        return ks, vs, k2s

    def __delitem__(self, key):
        del self.data_dict[key]

    def __getitem__(self, item):
        return self.data_dict[item]

    def pack(self, ks, vs, k2s=None, trans=True):
        self.data_dict = pack(ks=ks, vs=vs, k2s=k2s, trans=trans)
        return self.data_dict

    def transform(self, structure: List[Structure], **kwargs) -> List[Data]:
        """
        Transform data.

        Args:
            structure:(list) Preprocessing of samples need to transform to Graph.
            state_attr: (list)
                preprocessing of samples need to add to Graph.
            energy: (list)
                Target to train against (the same size with structure)

            **kwargs:

        Returns:
            list of graphs:
                List of dict

        """
        assert isinstance(structure, Iterable)
        if hasattr(structure, "__len__"):
            assert len(structure) > 0, "Empty input data!"

        if isinstance(structure, np.ndarray):
            structure = structure.tolist()

        le = len(structure)

        for i in kwargs.keys():
            if kwargs[i] is None:
                kwargs[i] = [kwargs[i]] * len(structure)
            elif not isinstance(kwargs[i], Iterable):
                kwargs[i] = [kwargs[i]] * len(structure)

        try:
            kw = [{k: v[i] for k, v in kwargs.items()} for i in range(le)]
        except IndexError as e:
            print(e)
            raise IndexError("Make sure the other parameters such as energy and state_attr"
                             " are with same number (length) of structure.")

        iterables = zip(structure, kw)

        if not self.batch_calculate:
            res = parallelize(self.n_jobs, self._wrapper, iterables, tq=self.tq, respective=True,
                              respective_kwargs=True, desc="Transforming", mode="j", )

        else:
            res = batch_parallelize(self.n_jobs, self._wrapper, iterables, respective=True,
                                    respective_kwargs=True, mode="j",
                                    tq=self.tq, batch_size=self.batch_size, desc="Transforming")

        ret, self.support_ = zip(*res)

        ret = [i for i, j in zip(ret, self.support_) if j is True]

        return ret

    def check_dup(self, structure, file_names="composition_name"):
        """Check the names duplication"""
        names = [i.composition.reduced_formula for i in structure]
        if file_names == "composition_name" and len(names) != len(set(names)):
            raise KeyError("There are same composition name for different structure, "
                           "please use file_names='rank_number' "
                           "to definition or specific names list.")
        elif len(set(file_names)) == len(structure):
            return file_names
        elif file_names == "rank_number":
            return ["raw_data_{}".format(i) for i in range(len(structure))]
        else:
            return names

    def transform_by_support(self, *args):
        """Transform ext sequence by support."""
        res = []
        for arg in args:
            assert len(arg) == len(self.support_)
            rei = np.array(arg)[np.array(self.support_)]
            res.append(rei)
        return tuple(res)

    def _save(self, obj, name, root_dir=".") -> None:
        """Save."""
        torch.save(obj, os.path.join(root_dir, "raw", '{}.pt'.format(name)))

    def transform_and_save(self, structure: list, energy=None, state_attr=None, root_dir=".",
                           file_names="composition_name", save_mode="o", **kwargs):
        r"""Save the data to 'root_dir/raw' if save_mode="i", else 'root_dir', compact with InMemoryDatasetGeo"""
        raw_path = os.path.join(root_dir, "raw")
        if os.path.isdir(raw_path):
            rmtree(raw_path)
        os.makedirs(raw_path)

        result = self.transform(structure, energy=energy, state_attr=state_attr, **kwargs)

        print("Save raw files to {}.".format(raw_path))
        if save_mode in ["i", "r", "respective"]:
            fns = self.check_dup(structure, file_names=file_names)
            [self._save(i, j, root_dir) for i, j in zip(result, fns)]
        else:
            self._save(result, "raw_data", root_dir=root_dir)
        print("Done.")
        return result

    def transform_data_dict_and_save(self, store_root_dir=".", msg_name=None):
        r"""Save the data to 'root_dir/raw' if save_mode="i", else 'root_dir', compact with InMemoryDatasetGeo"""
        raw_path = os.path.join(store_root_dir, "raw")
        if os.path.isdir(raw_path):
            rmtree(raw_path)
        os.makedirs(raw_path)

        result = self.transform_data_dict(msg_name=msg_name)

        print("Save raw files to {}. >>>".format(raw_path))

        self._save(result, "raw_data", root_dir=store_root_dir)

        print("Done.")
        return result

    def _convert(self, data: Structure, energy=None, state_attr=None, **kwargs) -> Data:
        z = torch.tensor(list(data.atomic_numbers))
        cell = np.copy(data.lattice.matrix)
        pbc = torch.tensor(data.pbc).view(3, 1)
        pos = np.copy(data.cart_coords)
        cell = torch.unsqueeze(torch.from_numpy(cell).float(), 0)
        pos = torch.from_numpy(pos).float()
        natoms = pos.shape[0]
        if state_attr is not None:
            state_attr = torch.from_numpy(np.array(state_attr)).float()
            state_attr = state_attr.reshape(1, 1) if len(state_attr.shape) == 0 else state_attr
        if energy is not None:
            energy = torch.tensor(energy).float()

        if not self.just_predefined:
            for key, value in kwargs.items():
                try:
                    value = torch.from_numpy(np.array(value))
                    value = value.reshape(1, 1) if len(value.shape) == 0 else value
                    kwargs[key] = value
                except ValueError:
                    pass
            return Data(energy=energy, pos=pos, z=z, state_attr=state_attr, cell=cell, natoms=natoms,
                        pbc=pbc, **kwargs)
        else:
            return Data(energy=energy, pos=pos, z=z,
                        state_attr=state_attr, cell=cell, natoms=natoms, pbc=pbc, )


class PAddSAPymatgen(StructureToData):
    """
    Add state_attr.
    pymatgen's structure.num_sites, structure.ntypesp, structure.density, structure.volume"""

    def __init__(self, *args, prop_name=None, **kwargs):
        if prop_name is None:
            self.prop_name = ['num_sites', 'ntypesp', 'density', 'volume']
        else:
            self.prop_name = prop_name
        super(PAddSAPymatgen, self).__init__(*args, **kwargs)

    @staticmethod
    def _convert(data: Structure, **kwargs) -> Data:
        state_attr = torch.tensor([data.num_sites, data.ntypesp, data.density, data.volume]).float()
        return Data(state_attr=state_attr)


class PAddXPymatgen(StructureToData):
    """
    Add x.

    Get pymatgen element preprocessing.
    prop_name = [
    "atomic_radius",
    "atomic_mass",
    "number",
    "max_oxidation_state",
    "min_oxidation_state",
    "row",
    "group",
    "atomic_radius_calculated",
    "mendeleev_no",
    "average_ionic_radius",
    "average_cationic_radius",
    "average_anionic_radius",]
    """

    def __init__(self, *args, prop_name=None, **kwargs):
        super(PAddXPymatgen, self).__init__(*args, **kwargs)
        if prop_name is None:
            self.prop_name = ["atomic_mass", "average_ionic_radius", "average_anionic_radius",
                              "atomic_radius_calculated"]
        else:
            self.prop_name = prop_name
        self.da = [Element.from_Z(i) for i in range(1, 119)]
        self.da.insert(0, self.da[0])  # for start from 1

    def _convert(self, data: Structure, **kwargs) -> Data:
        x = torch.tensor([[getattr(self.da[i], pi) for pi in self.prop_name] for i in data.atomic_numbers]).float()
        return Data(x=x)


class PAddXASE(StructureToData):
    """
    Add x.

    Get pymatgen element preprocessing.
    prop_name = [
    'atomic_masses', 'covalent_radii'
    ]
    """

    def __init__(self, *args, prop_name=None, **kwargs):
        super(PAddXASE, self).__init__(*args, **kwargs)
        if prop_name is None:
            self.prop_name = ['atomic_masses', 'covalent_radii']
        else:
            self.prop_name = prop_name
        arrays = np.concatenate([getattr(ase_data, i).reshape(-1, 1) for i in self.prop_name], axis=1)
        self.arrays = torch.from_numpy(arrays).float()

    def _convert(self, data: Structure, **kwargs) -> Data:
        x = self.arrays[data.atomic_numbers, :]
        return Data(x=x)


class PAddXArray(StructureToData):
    """
    Add x by np.ndarray (2D).
    The array is insert in 0 position automatically in code. (padding_0=True)

    """

    def __init__(self, array: np.ndarray, *args, padding_0=True, **kwargs):
        super(PAddXArray, self).__init__(*args, **kwargs)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if padding_0:
            array = np.concatenate((array[0, :].reshape(1, -1), array), axis=0)  # add one line in 0.
        self.arrays = torch.from_numpy(array).float()

    def _convert(self, data: Structure, **kwargs) -> Data:
        x = self.arrays[data.atomic_numbers, :]
        return Data(x=x)


class PAddForce(StructureToData):
    """
    Add forces by np.array (2D).
    """

    def __init__(self, *args, check=True, **kwargs):
        super(PAddForce, self).__init__(*args, **kwargs)
        self.check = check

    def _convert(self, data: Structure, **kwargs) -> Data:
        if self.check:
            assert "forces" in kwargs
            forces = torch.from_numpy(kwargs["forces"]).float()
            return Data(forces=forces)


class PAddXEmbeddingDict(StructureToData):
    """
    Add x by dict.
    """

    def __init__(self, *args, dct: dict = None, **kwargs):
        super(PAddXEmbeddingDict, self).__init__(*args, **kwargs)

        if dct is None:
            from hpc2ml.data.embedding.continuous_embeddings import continuous as dct
        self.dct = dct

    def _convert(self, data: Structure, **kwargs) -> Data:
        x = torch.Tensor([self.dct[ai] for ai in data.atomic_numbers]).float()
        return Data(x=x)


def _re_pbc(pbc: Union[bool, List[bool], np.ndarray], return_type="bool"):
    if pbc is True:
        pbc = [1, 1, 1]
    elif pbc is False:
        pbc = [0, 0, 0]
    elif isinstance(pbc, Iterable):
        pbc = [1 if i is True or i == 1 else 0 for i in pbc]
    else:
        raise TypeError("Can't accept {}".format(pbc))
    if return_type == "bool":
        pbc = np.array(pbc) == 1
    else:
        pbc = np.array(pbc)
    return pbc


def _get_r_in_spheres(data, pbc=True, cutoff=5.0, numerical_tol=1e-6):
    if isinstance(data, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(data.lattice.matrix), dtype=float)
        if pbc is not False:
            pbc = _re_pbc(pbc, return_type="int")
        else:
            pbc = np.array([0, 0, 0])
    else:
        raise ValueError("structure type not supported")

    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(data.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    images = images.astype(np.int64)
    distances = distances.astype(np.float32)
    exclude_self = (distances > numerical_tol)
    # exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    center_indices = center_indices[exclude_self]
    neighbor_indices = neighbor_indices[exclude_self]
    distances = distances[exclude_self]
    images = images[exclude_self]
    return center_indices, neighbor_indices, distances, images


def _get_xyz_in_spheres(data, pbc=True, cutoff=5.0, numerical_tol=1e-6):
    if isinstance(data, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(data.lattice.matrix), dtype=float)
        if pbc is not False:
            pbc = _re_pbc(pbc, return_type="int")
        else:
            pbc = np.array([0, 0, 0])
    else:
        raise ValueError("structure type not supported")
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(data.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    images = images.astype(np.int64)
    distances = distances.astype(np.float32)
    exclude_self = (distances > numerical_tol)

    center_indices = center_indices[exclude_self]
    neighbor_indices = neighbor_indices[exclude_self]
    distances = distances[exclude_self]
    images = images[exclude_self]

    offset = np.dot(images, data.lattice.matrix)

    coords = cart_coords[neighbor_indices, :] - cart_coords[center_indices, :]
    xyz = offset + coords

    return center_indices, neighbor_indices, distances, xyz


class PAddPBCEdgeSole(StructureToData):
    """
    Add Edge index with PBC.

    (Not suggest in network, if the edge_index depend on network coefficient.)

    """

    def __init__(self, *args, cutoff: float = 5.0,
                 numerical_tol: float = 1e-6,
                 pbc=True, **kwargs):
        """
        Get graph representations from structure within cutoff.

        Args:
            structure (pymatgen Structure or molecule)
            cutoff (float): cutoff radius
            numerical_tol (float): numerical tolerance
            nn_strategy(str,None):not used
        """
        super(PAddPBCEdgeSole, self).__init__(*args, **kwargs)
        self.numerical_tol = numerical_tol
        self.pbc = pbc
        self.cutoff = cutoff

    def _convert(self, data: Structure, **kwargs) -> Data:
        center_indices, neighbor_indices, distances, cell_offsets = _get_r_in_spheres(data, pbc=self.pbc,
                                                                                      cutoff=self.cutoff,
                                                                                      numerical_tol=self.numerical_tol)

        assert len(set(center_indices)) == data.num_sites, f"Some atom are independent at cutoff: {self.cutoff}"

        edge_index = torch.vstack([torch.from_numpy(neighbor_indices),
                                   torch.from_numpy(center_indices)])

        cell_offsets = torch.from_numpy(cell_offsets).float()

        return Data(edge_index=edge_index, cell_offsets=cell_offsets)


class PAddPBCEdgeDistance(StructureToData):
    """
    Add Edge with PBC.
    (Not suggest in newwork to calculate gradient of pos.)

    """

    def __init__(self, *args, cutoff: float = 5.0,
                 numerical_tol: float = 1e-6,
                 pbc=True, **kwargs):
        """
        Get graph representations from structure within cutoff.

        Args:
            structure (pymatgen Structure or molecule)
            cutoff (float): cutoff radius
            numerical_tol (float): numerical tolerance
            nn_strategy(str,None):not used
        Returns:
            center_indices, neighbor_indices, distances
        """
        super(PAddPBCEdgeDistance, self).__init__(*args, **kwargs)
        self.numerical_tol = numerical_tol
        self.pbc = pbc
        self.cutoff = cutoff

    def _convert(self, data: Structure, **kwargs) -> Data:
        center_indices, neighbor_indices, distances, cell_offsets = _get_r_in_spheres(data, pbc=self.pbc,
                                                                                      cutoff=self.cutoff,
                                                                                      numerical_tol=self.numerical_tol)

        assert len(set(center_indices)) == data.num_sites, f"Some atom are independent at cutoff: {self.cutoff}"

        edge_index = torch.vstack([torch.from_numpy(neighbor_indices),
                                   torch.from_numpy(center_indices)])
        edge_weight = torch.from_numpy(distances).float()

        cell_offsets = torch.from_numpy(cell_offsets).float()

        return Data(edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_weight.view(-1, 1),
                    cell_offsets=cell_offsets)


class PAddPBCEdgeXYZ(StructureToData):
    """
    Add edge xyz.
    (Not suggest in network, if calculate gradient of pos.)

    Get pymatgen element preprocessing.
    prop_name = [
    'vdw_radii', 'reference_states', 'atomic_masses', 'covalent_radii'
    ]
    """

    def __init__(self, *args, cutoff: float = 5.0,
                 numerical_tol: float = 1e-6,
                 pbc=True, **kwargs):
        """
        Get graph representations from structure within cutoff.

        Args:
            structure (pymatgen Structure or molecule)
            cutoff (float): cutoff radius
            numerical_tol (float): numerical tolerance
            nn_strategy(str,None):not used
        Returns:
            center_indices, neighbor_indices, distances
        """
        super(PAddPBCEdgeXYZ, self).__init__(*args, **kwargs)
        self.numerical_tol = numerical_tol
        self.pbc = pbc
        self.cutoff = cutoff

    def _convert(self, data: Structure, **kwargs) -> Data:
        center_indices, neighbor_indices, distances, distances_vec = _get_xyz_in_spheres(data, pbc=self.pbc,
                                                                                         cutoff=self.cutoff,
                                                                                         numerical_tol=self.numerical_tol)

        assert len(set(center_indices)) == data.num_sites, f"Some atom are independent at cutoff: {self.cutoff}"

        assert center_indices.shape == neighbor_indices.shape
        edge_index = torch.vstack([torch.from_numpy(center_indices),
                                   torch.from_numpy(neighbor_indices)])
        edge_weight = torch.from_numpy(distances).float()
        distances_vec = torch.from_numpy(distances_vec).float()

        return Data(edge_index=edge_index, edge_weight=edge_weight, edge_attr=distances_vec)


def voigt_6_to_full_3x3_stress(stress_vector):
    """
    Form a 3x3 stress matrix from a 6 component vector in Voigt notation
    """
    s1, s2, s3, s4, s5, s6 = np.transpose(stress_vector)
    return np.transpose([[s1, s6, s5],
                         [s6, s2, s4],
                         [s5, s4, s3]])


def full_3x3_to_voigt_6_stress(stress_matrix):
    """
    Form a 6 component stress vector in Voigt notation from a 3x3 matrix
    """
    stress_matrix = np.asarray(stress_matrix)
    return np.transpose([stress_matrix[..., 0, 0],
                         stress_matrix[..., 1, 1],
                         stress_matrix[..., 2, 2],
                         (stress_matrix[..., 1, 2] +
                          stress_matrix[..., 1, 2]) / 2,
                         (stress_matrix[..., 0, 2] +
                          stress_matrix[..., 0, 2]) / 2,
                         (stress_matrix[..., 0, 1] +
                          stress_matrix[..., 0, 1]) / 2])


class PAddStress(StructureToData):
    """
    Add stress."""

    def __init__(self, *args, stress_max: float = 15.0,
                 stress_min: float = -15.0, voigt_6=False, check: bool=True,
                 **kwargs):

        super(PAddStress, self).__init__(*args, **kwargs)

        self.stress_max = stress_max
        self.stress_min = stress_min
        self.voigt_6 = voigt_6
        self.check = check

    def _convert(self, data: Structure, **kwargs) -> Data:
        if self.check:
            assert "stress" in kwargs
            stress = np.array(kwargs["stress"])
            if self.voigt_6:
                if stress.shape == (3, 3):
                    stress = full_3x3_to_voigt_6_stress(stress)
                stress = torch.from_numpy(stress.reshape(1, -1)).float()
            else:
                if stress.shape == (6,):
                    stress = voigt_6_to_full_3x3_stress(stress)
                stress = torch.unsqueeze(torch.from_numpy(stress), dim=0)

            if torch.max(stress) > self.stress_max or torch.min(stress) < self.stress_min:
                raise ValueError("Bad structure with large stress out of range.")

            return Data(stress=stress)


class PAddFracCoords(StructureToData):
    """
    Add frac_coords."""

    @staticmethod
    def _convert(data: Structure, **kwargs) -> Data:
        frac_pos = torch.from_numpy(data.frac_coords)
        return Data(frac_pos=frac_pos)


class PAtomNumToType(StructureToData):
    num_types: int
    chemical_symbol_to_type: Optional[Dict[str, int]]
    type_names: List[str]
    _min_Z: int

    def __init__(self, *args, type_names: Optional[List[str]] = None,
                 chemical_symbol_to_type: Optional[Dict[str, int]] = None,
                 chemical_symbols: Optional[List[str]] = None, **kwargs):
        super(PAtomNumToType, self).__init__(*args, **kwargs)

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
            range(len(self.chemical_symbol_to_type)))
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
            ase_data.atomic_numbers[sym] for sym in self.chemical_symbol_to_type]
        self._min_Z = min(valid_atomic_numbers)
        self._max_Z = max(valid_atomic_numbers)
        Z_to_index = torch.full(
            size=(1 + self._max_Z - self._min_Z,), fill_value=-1, dtype=torch.long)
        for sym, type in self.chemical_symbol_to_type.items():
            Z_to_index[ase_data.atomic_numbers[sym] - self._min_Z] = type
        self._Z_to_index = Z_to_index
        self._index_to_Z = torch.zeros(
            size=(len(self.chemical_symbol_to_type),), dtype=torch.long)
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

    def _convert(self, data: Structure, **kwargs) -> Data:
        atomic_numbers = torch.tensor(list(data.atomic_numbers))
        zt = self._transform(atomic_numbers)
        return Data(zt=zt)

    def _transform(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(atomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!")

        return self._Z_to_index.to(device=atomic_numbers.device)[
            atomic_numbers - self._min_Z]
