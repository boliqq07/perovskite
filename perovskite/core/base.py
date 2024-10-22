# -*- coding: utf-8 -*-
# @Time  : 2023/4/19 17:10
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import copy
from typing import Tuple, List

import ase
from ase import Atoms
from ase.spacegroup import get_spacegroup

from perovskite.core.general_rotate import rotate_times, rotate_by_angle
from perovskite.core.general_standard import perovskite_prim_221, perovskite_conv_221, \
    perovskite_prim_62, perovskite_conv_127, perovskite_prim_127, perovskite_conv_62
from perovskite.core.general_substitute import sub_atoms
from perovskite.core.rotate_substitude import SubstituteWithRotatedB


class Perovskite(Atoms):
    """Specialized custom classes for Perovskite materials."""

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
        """From Atoms to Perovskite."""
        atomss = cls(cell=atoms.cell, pbc=atoms.pbc, info=atoms.info,
                     celldisp=atoms._celldisp.copy())

        atomss.arrays = {}
        for name, a in atoms.arrays.items():
            atomss.arrays[name] = a.copy()
        atomss.constraints = copy.deepcopy(atoms.constraints)
        return atomss

    def to_atoms(self) -> Atoms:
        """From Perovskite to Atoms."""
        atoms = Atoms(cell=self.cell, pbc=self.pbc, info=self.info,
                      celldisp=self._celldisp.copy())

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        atoms.constraints = copy.deepcopy(self.constraints)
        return atoms

    def rotate_by_angle(self, angle: int = 90, axis: Tuple[Tuple, int] = ((1, 0, 0),), center=(0, 0, 0)) -> List[
        Atoms]:
        """Rotate atoms self by angle along x, y, z axis"""
        return rotate_by_angle(self.copy(), angle, axis, center=center)

    def rotate_by_times(self, times: int = 4, axis: Tuple[Tuple, int, str] = ((1, 0, 0),)) -> List[Atoms]:
        """Rotate atoms self by times along x, y, z axis"""
        return rotate_times(self.copy(), times, axis)

    @classmethod
    def from_standard(cls, template, a_atom="Ca", b_atom="Ti", c_atom="O", a=7, alpha=1.0,
                      move=0.04, cycle=0.04, size=(1, 1, 1), **kwargs) -> "Perovskite":
        """
        Get perovskite by standard.

        Args:
            template: (str), perovskite type, available in ("prim_221","conv_221","prim_127","conv_127","prim_62","conv_62")
            a_atom: (str), a atom
            b_atom: (str), b atom
            c_atom: (str), c atom
            a: (str), a axis length of cell
            alpha: (float), c / a or b / a axis ratio
            move: (float), distortion of tetrahedron
            cycle: (float), distortion of tetrahedron
            size: (tuple), supercell site, default 111
            **kwargs:

        Returns:
            perovskite: (Perovskite), new object.

        """

        mp = {
            "prim_221": perovskite_prim_221,
            "conv_221": perovskite_conv_221,
            "prim_127": perovskite_prim_127,
            "conv_127": perovskite_conv_127,
            "prim_62": perovskite_prim_62,
            "conv_62": perovskite_conv_62
        }

        func = mp[template]
        atoms = func(a_atom=a_atom, b_atom=b_atom, c_atom=c_atom, a=a, alpha=alpha,
                     move=move, cycle=cycle, size=size, **kwargs)
        return cls(symbols=None, positions=atoms.positions, numbers=atoms.numbers,
                   cell=atoms.cell, pbc=atoms.pbc, celldisp=atoms._celldisp,
                   calculator=atoms.calc, )

    def sub_atoms(self, a: str, b: [str, ase.Atom, ase.Atoms], index=None) -> List[Atoms]:
        """Rotate atoms by times along x, y, z axis"""
        return sub_atoms(self.copy(), a, b, index)

    def substitute_with_rotated_b(self, old: str, new: [str, ase.Atom, ase.Atoms], index=None,
                                  strategy="auto", space_group_number=None, symprec=1e-5,
                                  ) -> "Perovskite":
        """
        Substitute atoms in atoms.

        such as: ABX3 -> ACX3

        such as: ABX3 -> ABO3

        such as: ABX3 -> (CH3NH3)BX3


        Args:
            old: (str,) old atom name.
            new: (str, ase.Atom, ase.Atoms), new atoms.
            index: tuple, part of index of all `a` atom (from 0 to N_a - 1). if index is None, substitute all `a` atom.
            strategy: ("same", "mirror", "random", "auto"), strategy to deal with b atoms before substitute.
            space_group_number: (float, "auto"), space_group_number
            symprec: float, tolerance to find space_group_number, if space_group_number="auto"

        Returns:


        """
        if space_group_number is None:
            sg = get_spacegroup(self, symprec=symprec)
            space_group_number = sg.no

        swrb = SubstituteWithRotatedB(strategy=strategy, space_group_number=space_group_number)

        return swrb.sub_atoms(self.copy(), a=old, b=new, index=index)

    def to_structure(self):
        from pymatgen.io.ase import AseAtomsAdaptor
        aaa = AseAtomsAdaptor()
        return aaa.get_structure(self)

    def show(self):
        """
        Plot by ase.
        """
        from ase.visualize import view
        view(self)

    def view(self):
        """The same as show. plot by ase."""
        self.show()

    @staticmethod
    def T_f(rA, rB, rX):
        """
        公差系数
        """
        t_f = (rA + rX) / (1.414 * (rB + rX))
        return t_f

    @staticmethod
    def O_f(rB, rX):
        """
        八面体因子
        """
        o_f = rB / rX
        return o_f