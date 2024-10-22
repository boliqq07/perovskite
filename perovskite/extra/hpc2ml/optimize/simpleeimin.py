import warnings
from collections import deque

import numpy as np

from hpc2ml.optimize.basegpmin import BaseGPMin


class SimpleEIGPMin(BaseGPMin):
    """EffectiveImprovement-GPMin"""
    def __init__(self, atoms, **kwargs):
        super(SimpleEIGPMin, self).__init__(atoms, **kwargs)

    def step(self, f=None):

        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()

        fc = self.force_consistent
        r0 = atoms.get_positions().reshape(-1)
        e0 = atoms.get_potential_energy(force_consistent=fc)
        self.update(r0, e0, f)

        r_add = None
        e_add = None
        f_add = None

        count = 0

        self.es = deque([e0], maxlen=3)

        while len(self.es) < 3 or np.std(self.es) > 0.01:  # 0.01 eV energy error

            r_, e_, f_ = self.resample_r_e_f(20,
                                             random_factor=0.01,  # this is fixed, 0.01 is best when test.
                                             )  # train

            if r_add is None:
                pass
            else:
                # Active learning iteration (add the sample with good performance).
                r_ = np.concatenate((r_, r_add), axis=0)
                e_ = np.concatenate((e_, e_add), axis=0)
                f_ = np.concatenate((f_, f_add), axis=0)

            self.update_batch(r_, e_, f_)  # train

            random_factor = max(0.05 * 1/(self.nsteps+1), 0.001)  # decrease the degree of position change
                                                                  # with step.

            r_ = self.resample_r(20, random_factor=random_factor)

            res = self.predict_mean_std_all(r_)

            ei = self.ei_balance(res)

            # ei = self.ei_balance(res, loss_coef=1-0.02*self.nsteps)
            # decrease the forces weight with step.
            # this should be just used for forces with
            index_ = np.argsort(-ei)[:5]
            r_add = r_[index_]

            r_add, e_add, f_add = self.r_to_e_f(r_add)

            index_min = np.argmin(e_add)  # best
            e1 = e_add[index_min]
            r1 = r_add[index_min]
            f1 = f_add[index_min]

            # if e1 < self.es[-1]+0.00001:
            if e1 < self.es[-1]:
                self.atoms.set_positions(r1.reshape(-1, 3))
                self.es.append(float(e1))

            self.function_calls += 1
            self.force_calls += 1

            if self.converged(f1.reshape(-1, 3)):
                break

            count += 1
            if count == 20:
                warnings.warn("A descent model could not be built")
                break

            # print(count, self.es[-1], e1)


        self.dump()



