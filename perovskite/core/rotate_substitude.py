# -*- coding: utf-8 -*-

# @Time  : 2023/4/19 17:23
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import ase
import numpy as np
from ase import Atom, Atoms
from ase.build import molecule
from ase.visualize import view
from numpy import random
from numpy.linalg import norm


class RotateSubstitute:
    """Substitute atoms in atoms.

    such as: ABX3 -> ACX3

    such as: ABX3 -> ABO3

    such as: ABX3 -> (CH3NH3)BX3

    """

    def __init__(self, strategy="same", **kwargs):
        assert strategy in ["same", "mirror", "random", "auto"]

        if strategy == "auto":
            assert "space_group_number" in kwargs

        self.strategy = strategy
        self.kwargs = kwargs

    def sub_atoms(self, atoms: ase.Atoms, a: str, b: [str, ase.Atom, ase.Atoms], index=None):
        """Substitute atom/atoms."""
        atoms = atoms.copy()
        if a not in atoms.symbols.species():
            raise TypeError("Cant find the a element {}".format(a))

        indexs = atoms.symbols.indices()[a]

        if isinstance(index, (list, tuple)):
            try:
                indexs = indexs[tuple(index)]
            except IndexError:
                raise TypeError("Number of index:{} is out of number of a atoms {}".format(len(index), len(indexs)))

        if isinstance(b, ase.Atoms):

            bbs = self.get_rotate_atoms_list(b, atoms=atoms, index=indexs)

            for indi, bb in zip(indexs, bbs):
                atoms.extend(bb)

            del atoms[indexs]
        elif isinstance(b, str) or isinstance(b, Atom):
            atoms.symbols[indexs] = b
        else:
            raise TypeError("b is str element name or ase.Atoms")

        return atoms

    def get_rotate_atoms_list(self, b, atoms=None, index=None):
        """rotate with strategy."""
        b = b.copy()

        if self.strategy == "same":
            bs = self._same(b, atoms=atoms, index=index, **self.kwargs)
        elif self.strategy == "auto":
            bs = self._auto(b, atoms=atoms, index=index, **self.kwargs)
        elif self.strategy == "random":
            bs = self._random(b, atoms=atoms, index=index, **self.kwargs)
        elif self.strategy == "mirror":
            bs = self._mirror(b, atoms=atoms, index=index, **self.kwargs)
        else:
            raise NotImplementedError

        bs = self._process(bs, atoms=atoms, index=index)
        return bs

    def _process(self, bs, atoms=None, index=None):
        """set the position for each b atoms"""
        for indi, b in zip(index, bs):
            b.center()
            b.set_pbc(False)
            b.set_positions(atoms.get_positions()[indi] + b.get_positions())
        return bs

    def _random(self, b: [str, ase.Atom, ase.Atoms], atoms=None, index=None, random_state=None, **kwargs):

        """Substitute atom/atoms in random direction."""

        random.seed(random_state)
        bs = []
        for _ in index:
            bb = b.copy()
            angle = random.randint(1, 360)
            axi = random.random(size=3)
            bb.rotate(a=angle, v=axi)
            bs.append(bb)
        return bs

    def _same(self, b: [str, ase.Atom, ase.Atoms], atoms=None, index=None, angle=0, v="x", **kwargs):

        """Substitute atom/atoms in same direction."""
        bs = []
        for _ in index:
            bb = b.copy()
            bb.rotate(a=angle, v=v)
            bs.append(bb)
        return bs

    def _nothing(self, b: [str, ase.Atom, ase.Atoms], atoms=None, index=None, **kwargs):

        """do nothing."""
        bs = []
        for _ in index:
            bb = b.copy()
            bs.append(bb)
        return bs

    def _auto(self, b: [str, ase.Atom, ase.Atoms], atoms: Atoms = None, index=None, **kwargs):

        """judge auto by space_group_number."""
        space_group_number = kwargs["space_group_number"]
        if space_group_number == 221:
            return self._nothing(b, atoms=atoms, index=index, **kwargs)

        elif space_group_number in [62, 127]:
            try:
                abc_ = atoms.cell.cellpar()[:3]
                c_ = sum(max(abc_) - abc_) - sum(min(abc_) - abc_)
                if c_ >= 0:
                    axi = np.argmax(abc_)
                else:
                    axi = np.argmin(abc_)
            except TypeError:
                axi = 2

            positions = atoms.get_scaled_positions()[index]

            tol = kwargs["tol"] if "tol" in kwargs else 0.1

            index_group = _merge(positions[:, axi], tol=tol)
            index_groupd = [j for i, j in enumerate(index_group) if i % 2 == 0]
            index_groupd = [j for i in index_groupd for j in i]

            if "v" not in kwargs:
                v = {2: "z", 1: "y", 0: "x"}[axi]
            else:
                v = kwargs["v"]

            bs = []
            for i in range(len(index)):
                bb = b.copy()

                if i in index_groupd:
                    bb.rotate(a=180, v=v)
                if i % 2 == 1:
                    bb.rotate(a=90, v=(1, 1, 0))
                bs.append(bb)
            return bs

    def _mirror(self, b: [str, ase.Atom, ase.Atoms], atoms: Atoms, index: np.ndarray, angle=180, v="z", **kwargs):

        """Substitute atom/atoms in same direction."""

        if len(index) == 1:
            return self._nothing(b, atoms=atoms, index=index, **kwargs)

        positions = atoms.get_scaled_positions()[index]

        distance = norm(positions, axis=1)

        tol = kwargs["tol"] if "tol" in kwargs else 0.1

        index_group = _merge(distance, tol=tol)
        index_groupd = [j for i, j in enumerate(index_group) if i%2==0]
        index_groupd = [j for i in index_groupd for j in i]

        bs = []
        for i in range(len(index)):
            bb = b.copy()
            if i in index_groupd:
                bb.rotate(a=angle, v=v)
            bs.append(bb)
        return bs


def _merge(distance, tol=0.1):
    distance_index = np.argsort(distance)
    distance = np.sort(distance)
    i = 0
    group = [[0, ]]
    while i < len(distance) - 1:
        if abs(distance[i] - distance[i + 1]) < tol:
            group[-1].append(i + 1)
        else:
            group.append([i + 1])
        i += 1

    index_group = [[distance_index[ii] for ii in i] for i in group]

    return index_group


if __name__ == "__main__":
    from .general_standard import perovskite_prim_221, perovskite_conv_221, perovskite_prim_62, \
        perovskite_conv_62
    atomss = perovskite_conv_62(a_atom="Ba", b_atom="Ti", c_atom="O", a=7, size=(1, 1, 1))
    # atomss = perovskite_conv_221(a_atom="Ba", b_atom="Ti", c_atom="O", a=7, size=(1, 1, 1))
    no = molecule('HCF3')
    no.get_scaled_positions()
    view(atomss)
    att =  RotateSubstitute(strategy="auto",space_group_number=62)
    # att =  BaseSubstitute(strategy="mirror",angle=90, v="x")
    atoms =att.sub_atoms(atomss, "Ba", no, index=None)
    view(atoms)