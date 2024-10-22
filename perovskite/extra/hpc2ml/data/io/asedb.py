"""
This part sparse db (ase) data.

Examples:
    >>> dict_data = sparse_ase(db="data.db")

Examples:
    >>> from hpc2ml.data.structuretodata import StructureToData
    >>> addsap = StructureToData()
    >>> source_path = "."
    >>> res = addsap.sparse_source_data(source_file="data.db", source_path=source_path, fmt="ase",)
"""

from typing import Dict, Any, Union

from ase.db.core import reserved_keys, connect
from ase.db.jsondb import JSONDatabase
from ase.db.row import AtomsRow
from ase.db.sqlite import SQLite3Database

from hpc2ml.data.io.ase import aaa


def db_row2dct(row: AtomsRow
               ) -> Dict[str, Any]:
    """Convert row to dict."""
    temp = {}
    temp.update(row.data)
    temp.update(row.key_value_pairs)
    dct = {i: row.__dict__[i] for i in reserved_keys if i in row.__dict__}
    temp.update(dct)
    atoms = row.toatoms()
    st = aaa.get_structure(atoms)
    temp.update({"structure": st})
    id_str = temp.get("id")
    return {id_str: temp}


def sparse_ase(db: Union[SQLite3Database, JSONDatabase, str]) -> Dict[str, Any]:
    """Sparse ase data."""
    db = connect(db)
    dct = {}
    for row in db.select():
        dct.update(db_row2dct(row))
    return dct
