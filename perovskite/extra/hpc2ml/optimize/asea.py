# -*- coding: utf-8 -*-
import pandas as pd
import pymatgen.core
# @Time  : 2023/1/31 16:55
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
from ase import Atoms
# from ase.optimize import BFGS, GPMin
from ase.calculators.emt import EMT
import numpy as np
from hpc2ml.data.io import aaa
from pymatgen.core import Structure

from hpc2ml.data.structuretodata import StructureToData, PAddXArray, PAddPBCEdgeDistance

from hpc2ml.nn.cggru import CGGRU

from hpc2ml.optimize.calculator import GNNEICalculater
from hpc2ml.optimize.simpleeimin import SimpleEIGPMin
from hpc2ml.optimize.regpmin import ResampleGPMin
from hpc2ml.optimize.eimin import EIMin
from hpc2ml.optimize.strengthengpmin import StrengthenGPMin
from hpc2ml.optimize.basegpmin import BaseGPMin

d = 0.9575
t = np.pi / 180 * 104.51
water = Atoms('H2O',
              positions=[(d, 0, 0),
                         (d * np.cos(t), d * np.sin(t), 0),
                         (0, 0, 0)],
              calculator=EMT())

dyn = BaseGPMin(water)
# dyn = ResampleGPMin(water)
# dyn = StrengthenGPMin(water,update_hyperparams=True, batch_size=10)
# dyn = EIMin(water)
# dyn = SimpleEIGPMin(water)
dyn.run(fmax=0.01)