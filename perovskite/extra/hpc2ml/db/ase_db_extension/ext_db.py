import os
import shutil
from pathlib import PurePath
from typing import Union, Tuple, List

from ase.db import connect
from ase.db.core import Database
from ase.db.row import AtomsRow

from .ext_row import check_name_tup, atoms_row_rename


def check_postfix(name: Union[Database, str, dict]):
    if isinstance(name, PurePath):
        name = str(name)

    if isinstance(name, dict):
        postfix = '.json'
    elif isinstance(name, str):
        postfix = os.path.splitext(name)[-1]
    else:
        from ase.db.jsondb import JSONDatabase
        from ase.db.sqlite import SQLite3Database

        if isinstance(name, JSONDatabase):
            postfix = '.json'
        elif isinstance(name, SQLite3Database):
            postfix = '.db'
        else:
            postfix = ''

    return postfix


def db_append(database1: Database, database2: Database, parallel=False) -> None:
    """Replace the in situ."""
    if parallel:
        with database1:
            with database2:
                for ar in database2.select():
                    id = database1.reserve(name=str(ar))
                    if id is None:
                        continue
                    database1.write(ar, ar.key_value_pairs, ar.data, id=id)
    else:
        with database1:
            with database2:
                for ar in database2.select():
                    database1.write(ar, ar.key_value_pairs, ar.data)


def db_remove(database: Database, sli: [slice, List, Tuple]) -> None:
    """Delete the in situ!!!!!,Keep the id in sli!!!."""
    if isinstance(sli, Tuple):
        assert len(sli) <= 3, 'just accept (start,stop,[step])'
        sli = slice(*sli)
    if isinstance(sli, slice):
        assert sli.start >= 1, "slice for database start from 1"
        if sli.step is None:
            sli = slice(sli.start, sli.stop, 1)
        sli_list = list(range(sli.start, sli.stop, sli.step))
    else:
        sli_list = sli
    with database:
        database.delete(sli_list)
    print(sli_list, "is deleted")


def db_slice(database: Database, sli: [slice, Tuple], new_database_name: str = None, paraller=False) -> Union[
    Database, List[AtomsRow]]:
    """Get slice of database, in new database, Keep the id in sli!!!"""
    if isinstance(database, str):
        database = connect(database)

    if isinstance(sli, Tuple):
        assert len(sli) <= 3, 'just accept (start,stop,[step])'
        sli = slice(*sli)

    assert isinstance(sli, slice) and sli.start >= 1, "slice for database start from 1"
    if sli.step is None:
        sli = slice(sli.start, sli.stop, 1)
    if isinstance(new_database_name, str):
        new_database_name = os.path.splitext(new_database_name)[0]
        new_database_name = "".join((new_database_name, "{}"))
        database1_ = database.__class__(new_database_name.format(check_postfix(database)))
        if paraller:
            with database:
                with database1_:
                    for i in range(sli.start, sli.stop, sli.step):
                        id = database1_.reserve(name=i)
                        if id is None:
                            continue
                        ar = database[i]
                        database1_.write(ar, ar.key_value_pairs, ar.data, id=id)
        else:
            with database:
                with database1_:
                    for i in range(sli.start, sli.stop, sli.step):
                        ar = database[i]
                        database1_.write(ar, ar.key_value_pairs, ar.data)
    else:
        database1_ = []
        with database:
            for i in range(sli.start, sli.stop, sli.step):
                database1_.append(database[i])
    return database1_


def db_cat(database1: Union[Database, str], database2: Union[Database, str], new_database_name: str) -> Database:
    """new database, in new database."""
    new_database_name = os.path.splitext(new_database_name)[0]
    new_database_name = "".join((new_database_name, "{}"))

    if isinstance(database1, Database):
        database1 = database1.filename

    if isinstance(database1, str):
        if isinstance(database2, str):
            database2 = connect(database2)

        assert os.path.isfile(database1)
        new = new_database_name.format(check_postfix(database1))
        shutil.copy(database1, new)
        new = connect(new)
        db_append(new, database2)
        return new

    else:
        # old deprecated
        if isinstance(database1, str):
            database1 = connect(database1)
        if isinstance(database2, str):
            database2 = connect(database2)
        database1_ = database1.__class__(new_database_name.format(check_postfix(database1)))

        with database1_:
            with database1:
                for ar in database1.select(selection=None):
                    database1_.write(ar, ar.key_value_pairs, ar.data)
            with database2:
                for ar in database2.select(selection=None):
                    database1_.write(ar, ar.key_value_pairs, ar.data)
        return database1_


def db_ids(database: Database, selection=None, **kwargs) -> Tuple:
    """Get ids.
    """
    ids = []
    with database:
        for row in database.select(selection, **kwargs):
            ids.append(row.id)

    t = ids[0]
    i = ids[0]
    pstr = "({}-".format(t)
    for i in ids:
        if i - t <= 1:
            pass
        else:
            pstr = pstr + "{},{}-".format(t, i)
        t = i
    pstr = pstr + "{})".format(i)
    print(pstr)
    return tuple(ids)


def db_rename(database: Database, name_pair=(("old_name1", "new_name1"),), check=True, selection=None,
              **kwargs) -> Database:
    if check:
        check_name_tup(name_pair)

    if os.path.splitext(database.filename)[1] == '.json':
        js = True
    else:
        js = False
    with database:
        for row in database.select(selection, **kwargs):
            new_row = atoms_row_rename(row, name_pair=name_pair, check=False)
            kvp = new_row.key_value_pairs
            data = new_row.get('data', {})

            if js:
                database._write(new_row, kvp, data, row.id)
            else:
                database._update(row.id, kvp, data)

    return database


def db_transform(database, new_database_name="new_database.json"):
    """Convert ase.json to ase.db, or inverse."""

    assert ".db" in new_database_name or ".json" in new_database_name
    if os.path.isfile(new_database_name):
        raise FileExistsError(new_database_name, "is exist, please change the `new_database_name`.")

    to_type = os.path.splitext(new_database_name)[1]

    assert to_type in [".json", ".db"]

    from ase.db.jsondb import JSONDatabase
    from ase.db.sqlite import SQLite3Database

    if isinstance(database, str):
        database = connect(database)
    database1_ = connect(new_database_name)

    if isinstance(database, JSONDatabase) and to_type == ".json":
        print("Initial type is json, do nothing.")
        return
    elif isinstance(database, SQLite3Database) and to_type == ".db":
        print("Initial type is db (SQLite3), do nothing.")
        return

    else:
        with database:
            with database1_:
                for ar in database.select():
                    database1_.write(ar, ar.key_value_pairs, ar.data)
        return database1_
