# -*- coding: utf-8 -*-

# @Time    : 2021/7/15 22:17
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

from typing import List, Tuple

from ase import Atoms


def rotate_times(atoms: Atoms, times: int = 4, axis: Tuple[Tuple, int, str] = ((1, 0, 0),)) -> List[Atoms]:
    """Rotate atoms by 360/times along x, y, z axis"""
    atss = []
    atoms.center()
    ang = int(360 / times)
    if axis is None:
        axis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if isinstance(axis, int) and axis < 3:
        axist = [[0, 0, 0]]
        axist[0][axis] = 1
        axis = axist
    for axi in axis:
        for i in range(times):
            atomc = atoms.copy()
            atomc.rotate(ang * i, v=axi)
            atss.append(atomc)

    return atss


def rotate_by_angle(atoms: Atoms, angle: int = 90, axis: Tuple[Tuple, int] = ((1, 0, 0),), center=(0, 0, 0)) -> List[
    Atoms]:
    """Rotate atoms by angle along x, y, z axis"""
    atss = []
    atoms.center()

    if axis is None:
        axis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if isinstance(axis, int) and axis < 3:
        axist = [[0, 0, 0]]
        axist[0][axis] = 1
        axis = axist
    for axi in axis:
        atomc = atoms.copy()
        atomc.rotate(angle, v=axi, center=center)
        atss.append(atomc)

    return atss
