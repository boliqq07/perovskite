import warnings

import numpy as np
from ase.io.jsonio import write_json
from ase.parallel import world
from scipy.optimize import minimize

from hpc2ml.optimize.basegpmin import BaseGPMin


class ResampleGPMin(BaseGPMin):
    def __init__(self, atoms, **kwargs):
        super(ResampleGPMin, self).__init__(atoms, **kwargs)

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

            r_, e_, f_ = self.resample_r_e_f(20, random_factor=0.005)  # predict by calc

            self.update_batch(r_, e_, f_)  # train gp model

            r1 = self.relax_model(r0)
            self.atoms.set_positions(r1.reshape(-1, 3))
            e1 = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            f1 = self.atoms.get_forces()

            self.function_calls += 1
            self.force_calls += 1

            if self.converged(f1):
                break

            count += 1
            if count == 10:
                warnings.warn("A descent model could not be built")
                break

        self.dump()











    