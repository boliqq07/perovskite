# -*- coding: utf-8 -*-
import copy

# @Time  : 2023/4/19 17:10
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from ase import Atoms
from .general_rotate import rotate_angle,rotate_times
from .general_standard import perovskite_prim_221,perovskite_conv_221,\
    perovskite_prim_62,perovskite_conv_127,perovskite_prim_127,perovskite_conv_62

from .general_substitute import sub_atoms
from .rotate_substitude import RotateSubstitute

class Perovskite(Atoms):
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None):
        super().__init__(symbols=symbols,
                 positions=positions, numbers=numbers,
                 tags=tags, momenta=momenta, masses=masses,
                 magmoms=magmoms, charges=charges,
                 scaled_positions=scaled_positions,
                 cell=cell, pbc=pbc, celldisp=celldisp,
                 constraint=constraint,
                 calculator=calculator,
                 info=info,
                 velocities=velocities)

    @classmethod
    def from_atoms(cls, atoms: Atoms):
        """Return a copy."""
        atomss = cls(cell=atoms.cell, pbc=atoms.pbc, info=atoms.info,
                     celldisp=atoms._celldisp.copy())

        atomss.arrays = {}
        for name, a in atoms.arrays.items():
            atomss.arrays[name] = a.copy()
        atomss.constraints = copy.deepcopy(atoms.constraints)
        return atomss

    def to_atoms(self)->Atoms:
        """Return a copy."""
        atoms = Atoms(cell=self.cell, pbc=self.pbc, info=self.info,
                      celldisp=self._celldisp.copy())

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        atoms.constraints = copy.deepcopy(self.constraints)
        return atoms