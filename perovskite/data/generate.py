# -*- coding: utf-8 -*-
# @Time  : 2023/5/5 13:21
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from ase import Atoms


def molecule_generator(molecule_data):
    for i, j in molecule_data.items():
        if isinstance(j["atom"], list):
            yield i, Atoms(symbols=j["atom"])
        else:
            yield i, j["atom"]