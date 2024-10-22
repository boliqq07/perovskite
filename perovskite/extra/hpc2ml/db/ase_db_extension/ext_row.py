# -*- coding: utf-8 -*-

# @Time     : 2021/8/29 18:55
# @Software : PyCharm
# @Author   : xxx

"""
This file is used to extension the compare AtomsRow(s).

Example

>>> db = connect("organometal.db")
>>> row1,row2 = db[1],db[2]
>>> atoms_row_matcher(row1,row2)
1.0 # the more close to 1, the more similar.

More compare:
    pymatgen/analysis/structure_matcher.py
    ase/utils/structure_comparator.py
"""

from collections import Iterable
from typing import Dict

import numpy as np
from ase.db.row import AtomsRow


def _equal(ar1: AtomsRow, ar2: AtomsRow, tol=2, resort=True):
    """Just sort site."""
    poa = np.round(ar1.positions, decimals=tol)
    pob = np.round(ar2.positions, decimals=tol)

    if resort:
        poa = np.sort(poa, axis=0)
        poa_index = np.argsort(poa, axis=0)
        pob = np.sort(pob, axis=0)
        pob_index = np.argsort(pob, axis=0)

        return (ar1.natoms == ar2.natoms and
                (poa == pob).all() and (ar1.numbers[poa_index] == ar2.numbers[pob_index]).all())
    else:
        return (ar1.natoms == ar2.natoms and
                (poa == pob).all() and (ar1.numbers == ar2.numbers).all())


def atoms_row_matcher_log(atomsrow1: AtomsRow, atomsrow2: AtomsRow, resort=False) -> Dict:
    """Check the log AtomsRow similarity"""
    result = {"atoms": _equal(atomsrow1, atomsrow2, tol=2, resort=resort)}
    for i in atomsrow1.key_value_pairs.keys() & atomsrow2.key_value_pairs.keys():
        result.update({i: atomsrow1.key_value_pairs[i] == atomsrow2.key_value_pairs[i]})
    for i in atomsrow1.key_value_pairs.keys() - atomsrow2.key_value_pairs.keys():
        result.update({i: False})
    for i in atomsrow2.key_value_pairs.keys() - atomsrow1.key_value_pairs.keys():
        result.update({i: False})
    for i in ['initial_magmoms', 'initial_charges', 'momenta', 'pbc']:
        h1, h2 = hasattr(atomsrow1, i), hasattr(atomsrow2, i)
        if h1 and h2:
            result.update({i: getattr(atomsrow1, i) == getattr(atomsrow2, i)})
        elif not h1 and not h2:
            pass
        else:
            result.update({i: False})

    for i in result.keys():
        if isinstance(result[i], np.ndarray):
            result[i] = np.all(result[i])

    return result


def atoms_row_matcher(atomsrow1: AtomsRow, atomsrow2: AtomsRow) -> bool:
    """Check the score AtomsRow similarity"""
    score = atoms_row_matcher_log(atomsrow1, atomsrow2)
    return np.mean(np.array(list(score.values())).astype(float))


class HashAtomsRow(AtomsRow):
    """Add equal method, the same site is the same atom."""

    @classmethod
    def from_atomsrow(cls, atomsrow: AtomsRow):
        """Return a copy."""
        atomss = cls(dct={"numbers": atomsrow.numbers})
        atomss.__dict__.update(atomsrow.__dict__)
        return atomss

    def to_atomsrow(self) -> AtomsRow:
        """Return a copy."""
        atomss = AtomsRow(dct={"numbers": self.numbers})
        atomss.__dict__.update(self.__dict__)
        return atomss

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, ar2: AtomsRow, tol=2, resort=True):
        poa = np.round(self.positions, decimals=tol)
        pob = np.round(ar2.positions, decimals=tol)

        if resort:
            poa = np.sort(poa, axis=0)
            poa_index = np.argsort(poa, axis=0)
            pob = np.sort(pob, axis=0)
            pob_index = np.argsort(pob, axis=0)

            return (self.natoms == ar2.natoms and
                    (poa == pob).all() and (self.numbers[poa_index] == ar2.numbers[pob_index]).all())
        else:
            return (self.natoms == ar2.natoms and
                    (poa == pob).all() and (self.numbers == ar2.numbers).all())


def check_name_tup(name_pair):
    assert isinstance(name_pair, Iterable)
    for i in name_pair:
        assert len(i) == 2
        for j in i:
            assert isinstance(j, str) and j not in ["data", "_data", "key_value_pairs", "_keys"]


def atoms_row_rename(atomsrow: AtomsRow, name_pair=(("old_name1", "new_name1"),), check=True):
    """Rename the name in data or key_value_pairs of row."""
    if check:
        check_name_tup(name_pair)

    kvp = atomsrow.key_value_pairs
    data = atomsrow.get('data', {})

    olds = []

    for o, n in name_pair:
        if o in kvp:
            kvp[n] = kvp[o]
            del kvp[o]

        elif o in data:
            data[n] = data[o]
            del data[o]
        olds.append(o)

    [delattr(atomsrow, i) for i in olds if hasattr(atomsrow, i)]
    atomsrow._keys = list(kvp.keys())
    atomsrow.__dict__.update(kvp)
    atomsrow._data = {}
    atomsrow._data.update(data)
    return atomsrow
