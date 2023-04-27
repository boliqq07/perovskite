# -*- coding: utf-8 -*-

# @Time    : 2021/7/16 12:02
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import ase
from ase import Atom


def sub_atoms(atoms: ase.Atoms, a: str, b: [str, ase.Atom, ase.Atoms], index=None):
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

        for indi in indexs:
            bc = b.copy()
            bc.center()
            bc.set_pbc(False)
            bc.set_positions(atoms.get_positions()[indi] + bc.get_positions())
            atoms.extend(bc)

        del atoms[indexs]
    elif isinstance(b, str) or isinstance(b, Atom):
        atoms.symbols[indexs] = b
    else:
        raise TypeError("b is str element name or ase.Atoms")

    return atoms
