# -*- coding: utf-8 -*-

# @Time    : 2021/7/18 23:54
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np

from ase import Atoms
from pymatgen.core import Element


def r_spheres_effective(atoms: Atoms):
    """
    For A site.
    This function just calculated with the ionic radii of pymatgen with  universe valence.
    but for some mulecule, the  valence are changed. Thus , For reference only.
    r_eff= rmass + rion, where rmass
        is defined as the distance between the centre of mass of the A‑site organic cation and the
        atom with the largest distance to the centre of mass (excluding hydrogen atoms), and rion
        is the corresponding ionic radius of the aforementioned atom. rXeff can be defined in a similar
        way to rAeff."""

    com = atoms.get_center_of_mass(scaled=False)
    atoms.set_center_of_mass(com, scaled=False)

    posi = atoms.positions
    distance = np.linalg.norm(posi, axis=1)
    drop_H = [i for i in range(len(atoms)) if atoms[i].number == 1]

    left = []
    for i in range(len(atoms)):
        if i in drop_H:
            pass
        else:
            left.append(i)

    left = np.array(left)

    distance = distance[left]
    index = np.argmax(distance)
    r_mass = np.max(distance)
    z = atoms[left[index]].number
    if z == 1:
        r_ion = 0.0
    elif z == 6:  # for C element, the C-4 use 177pm.
        r_ion = 1.77
    else:
        ele = Element.from_Z(z)
        r_ion_dict = ele.ionic_radii
        r_ion = list(r_ion_dict.values())[0]
    return r_mass + r_ion


def r_cylinders_effective(atoms: Atoms):
    """
    For C site.

    This function just calculated with the ionic radii of pymatgen with  universe valence.
    but for some mulecule, the  valence are changed. Thus , For reference only.

    return r_eff,r_h
    """

    com = atoms.get_center_of_mass(scaled=False)
    atoms.set_center_of_mass(com, scaled=False)

    drop_H = [i for i in range(len(atoms)) if atoms[i].number == 1]

    left = []
    for i in range(len(atoms)):
        if i in drop_H:
            pass
        else:
            left.append(i)
    left = np.array(left)

    positions = atoms.positions[:, np.newaxis, :]
    positions2 = atoms.positions[np.newaxis, :, :]
    posi = positions - positions2

    distance = np.linalg.norm(posi, axis=2)
    distance = distance[left][left]
    index = np.argmax(distance, axis=0)

    index = [np.argmax(index), max(index)]

    r_h = np.max(distance)

    for indexi in index:
        z = atoms[left[indexi]].number
        if z == 1:
            r_ion = 0.0
        elif z == 6:  # for C element, the C-4 use 177pm.
            r_ion = 1.77
        else:
            ele = Element.from_Z(z)
            r_ion_dict = ele.ionic_radii
            r_ion = list(r_ion_dict.values())[0]
        r_h += r_ion

    # part2

    com = atoms.get_center_of_mass(scaled=False)
    atoms.set_center_of_mass(com, scaled=False)

    posi = atoms.positions
    distance2 = np.linalg.norm(posi, axis=1)

    distance2 = distance2[left]
    index2 = np.argmax(distance2)

    orita = posi[left[index]]
    orita = orita[0] - orita[1]

    vec = posi[left[index2]]

    norm_orita = np.linalg.norm(orita)
    norm_vec = np.linalg.norm(vec)
    cos_angle = orita.dot(vec) / (norm_orita * norm_vec)
    sin_angle = (1 - cos_angle ** 2) ** 0.5
    r_mass = sin_angle * norm_vec

    z = atoms[left[index2]].number

    if z == 1:
        r_ion = 0.0
    elif z == 6:  # for C element, the C-4 use 177pm.
        r_ion = 1.77
    else:
        ele = Element.from_Z(z)
        r_ion_dict = ele.ionic_radii
        r_ion = list(r_ion_dict.values())[0]

    r_eff = r_mass + r_ion

    return r_eff, r_h


def get_all_ion_radii():
    for i in range(1, 100):
        ele = Element.from_Z(i)
        r_ion_dict = ele.ionic_radii
        r_ion = list(r_ion_dict.values())
        if len(r_ion) > 0:
            r_ion = r_ion[0]
        else:
            r_ion = np.NaN

        print(ele.symbol, r_ion)


if __name__ == "__main__":
    get_all_ion_radii()
