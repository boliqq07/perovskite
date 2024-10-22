# -*- coding: utf-8 -*-

# @Time     : 2021/8/29 18:55
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
"""Interface for ase.db and csv"""

import datetime
import json
import numbers
import os
import re
import warnings
from collections.abc import Callable
from typing import Dict, Union, Tuple, Any, List

import numpy as np
import pandas as pd
from ase.db import connect
from ase.db.core import Database, reserved_keys
from ase.db.row import AtomsRow, atoms2dict
from ase.io.jsonio import object_hook
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

reserved_keys_extend = reserved_keys

special_kv_keys = {'project', 'name', 'symmetry', 'space_group'}

word = re.compile('[_a-zA-Z][_0-9a-zA-Z]*$')

_aaa = AseAtomsAdaptor()


def _code_dct(dct: Dict) -> Dict:
    """Change the numpy as list for store."""
    for k, v in dct.items():

        if hasattr(v, 'todict'):
            d = v.todict()

            if not isinstance(d, dict):
                raise RuntimeError('todict() of {} returned object of type {} '
                                   'but should have returned dict'
                                   .format(v, type(d)))
            if hasattr(v, 'ase_objtype'):
                d['__ase_objtype__'] = v.ase_objtype
            dct[k] = d
        elif isinstance(v, np.ndarray):
            flatobj = v.ravel()
            if np.iscomplexobj(v):
                flatobj.dtype = v.real.dtype
            dct[k] = {'__ndarray__': (v.shape,
                                      v.dtype.name,
                                      flatobj.tolist())}
        elif isinstance(v, np.integer):
            dct[k] = int(v)
        elif isinstance(v, np.bool_):
            dct[k] = bool(v)
        elif isinstance(v, datetime.datetime):
            dct[k] = {'__datetime__': v.isoformat()}
        elif isinstance(v, complex):
            dct[k] = {'__complex__': (v.real, v.imag)}

        # elif isinstance(v, np.ndarray):
        #     dct[k] = v.tolist()
        elif isinstance(v, dict):
            dct[k] = _code_dct(v)
        elif isinstance(v, (list, tuple)):
            dct[k] = [_code_dct(i) if isinstance(i, dict) else v for i in v]

        else:
            dct[k] = v

    return dct


def db_row2dct(row: AtomsRow
               ) -> Dict[str, Any]:
    """Convert row to dict,and change the numpy as list for store."""
    temp = {}
    temp.update(row.data)
    temp.update(row.key_value_pairs)
    dct = {i: row.__dict__[i] for i in reserved_keys if i in row.__dict__}
    temp.update(dct)

    temp = _code_dct(temp)

    return temp


def _decode_dct(dct: Dict) -> Dict:
    """resume the list to np.ndarray."""
    for k, v in dct.items():
        if k == "unique_id":
            dct[k] = v

        elif isinstance(v, float):
            if np.isnan(v):
                dct[k] = None
            else:
                dct[k] = v
        else:
            try:
                if isinstance(v, str):
                    v = eval(v)
                if isinstance(v, (list, tuple)):
                    if not v:
                        dct[k] = None
                    else:
                        try:
                            nda = np.array(v)
                            if nda.dtype in (np.float64, np.int32) and nda.ndim >= 1:
                                dct[k] = nda
                            else:
                                dct[k] = [_decode_dct(i) for i in v]
                        except:
                            dct[k] = [_decode_dct(i) for i in v]

                elif isinstance(v, dict):
                    v = object_hook(v)
                    if isinstance(v, dict):
                        dct[k] = _decode_dct(v)
                    else:
                        dct[k] = v
                else:
                    dct[k] = v
            except (ValueError, NameError, TypeError, SyntaxError):
                pass
    return dct


def collect_dct3(dct: Dict, kv_names=None) -> Tuple:
    """Take part the dict to 3 part."""
    if "data" in dct:
        data = dct["data"]
        del dct["data"]
    else:
        data = {}

    if "key_value_pairs" in dct:
        key_value_pairs = dct["key_value_pairs"]
        del dct["key_value_pairs"]
    else:
        key_value_pairs = {}

    temp = {}

    temp.update(data)
    temp.update(key_value_pairs)
    temp.update(dct)

    new_dct, new_data, new_kv = {}, {}, {}

    kv_keys = special_kv_keys if kv_names is None else special_kv_keys & set(kv_names)

    for k, v in temp.items():
        if k in reserved_keys:
            new_dct.update({k: v})
        elif k in kv_keys:
            new_kv.update({k: v})
        elif not word.match(k) or not isinstance(v, (numbers.Real, np.bool_)):
            new_data.update({k: v})
        else:
            new_kv.update({k: v})

    new_dct["key_value_pairs"] = new_kv
    new_dct["data"] = new_data

    return new_dct, new_kv, new_data


def db_dct2row(dct: Dict, kv_names=None) -> [AtomsRow, Dict, Dict]:
    """Convert dict to row."""
    dct = _decode_dct(dct)
    new_dct, new_kv, new_data = collect_dct3(dct, kv_names=kv_names)
    atomsrow = AtomsRow(new_dct)
    return atomsrow, new_kv, new_data


def collect_dct1(dct: Dict) -> Dict:
    new_dct, new_kv, new_data = collect_dct3(dct)
    new_data.update(new_kv)
    new_data.update(new_dct)
    return new_data


def db_to_csv(database: Union[str, Database], csv_file_name=""):
    """Store the db to csv."""
    if isinstance(database, str):
        database = connect(database)

    data = []

    for row in database.select(selection=None):
        try:
            data.append(db_row2dct(row))
        except (KeyError, StopIteration, AttributeError) as e:
            print(e, row)

    data = pd.DataFrame.from_dict(data)

    if csv_file_name:
        data.to_csv("{}.csv".format(os.path.splitext(csv_file_name)[0]))
    elif csv_file_name == "":
        csv_file_name = os.path.split(database.filename)[-1]
        csv_file_name = os.path.splitext(csv_file_name)[0]
        data.to_csv("{}.csv".format(csv_file_name))
    else:
        return data


def db_from_ase_csv(csv_file: Union[str, pd.DataFrame], new_database_name="new_database.db", index_col=0,
                    pop_name: List[str] = None
                    ) -> Database:
    """Convert csv(with strand Atoms table) to ase.db.
    see also: db_to_csv"""
    if pop_name is not None:
        assert isinstance(pop_name, list)
    if new_database_name == "" and isinstance(csv_file, str):
        new_database_name = "".join((os.path.splitext(csv_file)[0], ".db"))
    assert ".db" in new_database_name or ".json" in new_database_name
    if os.path.isfile(new_database_name):
        raise FileExistsError(new_database_name, "is exist, please change the `new_database_name`.")
    database = connect(new_database_name)
    if isinstance(csv_file, str):
        file = pd.read_csv(csv_file, index_col=index_col)
    else:
        file = csv_file

    data = file.to_dict(orient="index")

    with database:
        for k in data.keys():
            atomsrow, new_kv, new_data = db_dct2row(data[k])
            if pop_name is not None:
                for i in pop_name:
                    if i in new_kv:
                        del new_kv[i]
                    elif i in new_data:
                        del new_data[i]
            database.write(atomsrow, key_value_pairs=new_kv, data=new_data)
    return database


def db_from_structure_csv(csv_file: Union[str, pd.DataFrame], new_database_name="new_database.db", index_col=0,
                          structure_index_name: str = None, fmt: Union[str, Callable] = "json",
                          pop_name: List[str] = None
                          ):
    """Convert csv to ase.db. The structure must be offered."""

    if isinstance(csv_file, str):
        file = pd.read_csv(csv_file, index_col=index_col)
    else:
        file = csv_file

    data = file.to_dict(orient="index")

    if new_database_name == "" and isinstance(csv_file, str):
        new_database_name = "".join((os.path.splitext(csv_file)[0], ".db"))
    assert ".db" in new_database_name or ".json" in new_database_name

    if structure_index_name and structure_index_name in file.columns:
        name = structure_index_name
    elif "structure" in file.columns:
        name = "structure"
    else:
        raise NameError("There must be structure index name.")

    database = db_from_structure_dict(data, new_database_name=new_database_name,
                                      structure_index_name=name, fmt=fmt, pop_name=pop_name)

    return database


def db_from_structure_json(json_file: Union[str, dict], new_database_name="new_database.db",
                           structure_index_name: str = "structure", fmt: Union[str, Callable] = "json",
                           pop_name: List[str] = None
                           ):
    """Convert csv to ase.db. The structure must be offered."""

    def read_json(i):
        f = open(i)
        entries = json.load(f)
        f.close()
        return entries

    if isinstance(json_file, str):
        data = read_json(json_file)
    else:
        data = json_file

    if new_database_name == "" and isinstance(json_file, str):
        new_database_name = "".join((os.path.splitext(json_file)[0], ".db"))
    assert ".db" in new_database_name or ".json" in new_database_name

    if structure_index_name is None:
        structure_index_name = "structure"
    name = structure_index_name

    database = db_from_structure_dict(data, new_database_name=new_database_name,
                                      structure_index_name=name, fmt=fmt, pop_name=pop_name)

    return database


# def fmt(st):
#     st.lattice._pbc=np.array([True,True,True])
#     try:
#         atoms = aaa.get_atoms(st)
#     except ValueError:
#         st.remove_site_property("selective_dynamics")
#         atoms = aaa.get_atoms(st)
#     return atoms


def db_from_structure_dict(data, new_database_name="new_database.db",
                           structure_index_name: str = None, fmt: Union[str, Callable] = "json",
                           pop_name: List[str] = None):
    """Convert csv to ase.db. The structure must be offered."""
    if pop_name is not None:
        assert isinstance(pop_name, list)

    assert ".db" in new_database_name or ".json" in new_database_name
    if os.path.isfile(new_database_name):
        raise FileExistsError(new_database_name, "is exist, please change the `new_database_name`.")

    if isinstance(data, list):
        number = range(len(data))
    else:
        number = data.keys()

    database = connect(new_database_name)

    with database:
        for k in number:
            try:
                datak = data[k]
                structure = datak.pop(structure_index_name)
                if isinstance(fmt, str):
                    st = Structure.from_str(structure, fmt=fmt)
                    atoms = _aaa.get_atoms(st)
                elif isinstance(fmt, Callable):
                    atoms = fmt(structure)
                dct = _decode_dct(datak)
                new_dct, new_kv, new_data = collect_dct3(dct)
                dct = atoms2dict(atoms)
                dct.update(new_dct)
                if pop_name is not None:
                    for i in pop_name:
                        if i in new_kv:
                            del new_kv[i]
                        elif i in new_data:
                            del new_data[i]

                database.write(dct, key_value_pairs=new_kv, data=new_data)
            except NameError as e:
                # except BaseException as e:
                warnings.warn("The {} sample is can't be analysis.".format(k))
                print(e)

    return database
