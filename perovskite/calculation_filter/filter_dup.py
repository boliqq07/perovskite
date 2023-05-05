# -*- coding: utf-8 -*-

# @Time    : 2021/7/15 22:34
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause


import copy
from typing import List

import ase
import numpy as np
from ase import Atoms


class CircaAtoms(Atoms):
    """check the 2 atoms is similar."""

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms):
        """Return a copy."""
        atomss = cls(cell=atoms.cell, pbc=atoms.pbc, info=atoms.info,
                     celldisp=atoms._celldisp.copy())

        atomss.arrays = {}
        for name, a in atoms.arrays.items():
            atomss.arrays[name] = a.copy()
        atomss.constraints = copy.deepcopy(atoms.constraints)
        return atomss

    def to_atoms(self) -> Atoms:
        """Return a copy."""
        atoms = Atoms(cell=self.cell, pbc=self.pbc, info=self.info,
                      celldisp=self._celldisp.copy())

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        atoms.constraints = copy.deepcopy(self.constraints)
        return atoms

    def __eq__(self, other, tol=2):

        if not isinstance(other, Atoms):
            return False

        a = self.arrays
        b = other.arrays

        poa = np.round(a['positions'], decimals=tol)
        pob = np.round(b['positions'], decimals=tol)

        poa = np.sort(poa, axis=0)
        poa_index = np.argsort(poa, axis=0)
        pob = np.sort(pob, axis=0)
        pob_index = np.argsort(pob, axis=0)
        return (len(self) == len(other) and
                (poa == pob).all()
                and (a['numbers'][poa_index] == b['numbers'][pob_index]).all())


def filter_dup(atomss: List[Atoms, CircaAtoms]) -> List[Atoms]:
    """Remove the same atoms."""
    temp = []
    for n, i in enumerate(atomss):
        mark_i = []
        for j in temp:
            if i == j:
                mark_i.append(True)
            else:
                mark_i.append(False)
        if not any(mark_i):
            temp.append(i)
    return temp
