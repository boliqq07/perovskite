import queue
from collections import deque

import numpy as np
import warnings

from scipy.optimize import minimize
from ase.parallel import world
from ase.io.jsonio import write_json
from ase.optimize.optimize import Optimizer
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior

from .basegpmin import BaseGPMin
from .gp import GaussianProcess


class StrengthenGPMin(BaseGPMin):
    def __init__(self, atoms, update_hyperparams=True, batch_size=10, **kwargs):
        super(StrengthenGPMin, self).__init__(atoms, update_hyperparams= update_hyperparams,
                                              batch_size=batch_size, **kwargs)

    def step(self, f=None):
        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()

        fc = self.force_consistent
        r0 = self.atoms.get_positions().reshape(-1)
        e0 = self.atoms.get_potential_energy(force_consistent=fc)
        self.update(r0, e0, f)

        r1 = self.relax_model(r0)
        self.atoms.set_positions(r1.reshape(-1, 3))
        e1 = self.atoms.get_potential_energy(force_consistent=fc)
        f1 = self.atoms.get_forces()
        self.function_calls += 1
        self.force_calls += 1
        count = 0

        while e1 >= e0:
            self.update(r1, e1, f1)

            random_factor = max(0.01 * 1 / (self.nsteps + 1),
                                0.001)  # decrease the degree of position change with step.

            r_ = self.resample_r(20, random_factor=random_factor)

            res = self.predict_mean_std_all(r_)

            ei = self.ei_balance(res)

            index_ = np.argsort(-ei)[:3]
            r_add = r_[index_]

            r_add, e_add, f_add = self.r_to_e_f(r_add)
            self.update_batch(r_add, e_add, f_add)  # train gp model

            r1 = self.relax_model(r0)
            self.atoms.set_positions(r1.reshape(-1, 3))
            e1 = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            f1 = self.atoms.get_forces()

            self.function_calls += 1
            self.force_calls += 1

            if self.converged(f1):
                break

            count += 1
            if count == 5:
                warnings.warn("A descent model could not be built")
                break

        self.dump()











    