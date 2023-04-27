# -*- coding: utf-8 -*-

# @Time    : 2021/7/16 11:14
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from typing import Union, List

import numpy as np
from ase import Atoms
from mgetool.tool import parallelize
from pymatgen.analysis.energy_models import EwaldElectrostaticModel
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

aaa = AseAtomsAdaptor()


def get_stability(structure: Union[Structure, Atoms]):
    """get energy by EwaldElectrostaticModel."""
    if isinstance(structure, Atoms):
        structure = aaa.get_structure(atoms=structure)
    eem = EwaldElectrostaticModel(acc_factor=8.0)
    try:
        return eem.get_energy(structure)
    except BaseException:
        return 10.0


def filter_stability(structure: List[Union[Structure, Atoms]], n_jobs=1, ):
    energies = parallelize(n_jobs, get_stability, structure, respective=False, tq=True,
                           batch_size='auto', mode="j")
    return np.array(energies)

