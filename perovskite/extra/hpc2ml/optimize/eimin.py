import warnings
from collections import deque

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.optimize.optimize import Optimizer

from hpc2ml.optimize.basegpmin import BaseGPMin


class EIMin(BaseGPMin):
    """EffectiveImprovementMin"""

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=None, converged_std="forces",
                 update_hyperparams=False, batch_size=2):

        self.function_calls = 1
        self.force_calls = 0
        self.x_list = []  # Training set features
        self.y_list = []  # Training set targets

        Optimizer.__init__(self, atoms, restart, logfile,
                           trajectory, master, force_consistent)

        if batch_size is None:
            self.nbatch = 1
        else:
            self.nbatch = batch_size

        self.converged_std = converged_std
        self.update_hp = update_hyperparams
        self.function_calls = 1
        self.force_calls = 0

    def update(self, r, e, f):
        """Update the PES

        Update the training set, the prior and the hyperparameters.
        Finally, train the model
        """
        # update the training set
        self.x_list.append(r)
        f = f.reshape(-1)
        y = np.append(np.array(e).reshape(-1), -f)
        self.y_list.append(y)
        if len(self.y_list) > 32:
            self.x_list = self.x_list[-32:]
            self.y_list = self.y_list[-32:]

    def update_batch(self, rs, es, fs):
        """Update the PES

        Update the training set, the prior and the hyperparameters.
        Finally, train the model
        """
        # update the training set
        [self.x_list.append(ri) for ri in rs]
        y = np.concatenate((np.array(es).reshape(-1, 1), -fs), axis=1)
        [self.y_list.append(yi) for yi in y]

        if len(self.y_list) > 32:
            self.x_list = self.x_list[-32:]
            self.y_list = self.y_list[-32:]

        if (self.update_hp and self.function_calls % self.nbatch == 0 and
                self.function_calls != 0):
            self.train()

    def train(self):
        """This train is on model."""
        if not hasattr(self.atoms.calc, "train"):
            return None
        atom_temp = self.atoms.copy()
        atom_temp.calc = self.atoms.calc

        atomss = []
        energys = []
        forcess = []
        for x, y in zip(self.x_list, self.y_list):
            pos = x
            atom_temp.set_positions(pos.reshape(-1, 3))
            energy = y[0]
            forces = y[1:].reshape(-1, 3)
            atomss.append(atom_temp)
            energys.append(energy)
            forcess.append(forces)

        self.atoms.calc.train(atomss, energy=energys, forces=forcess)

    def predict_mean_std(self, x):
        """In general, this part should use the values from vasp.
        But for simple, we just re-back the values from model."""
        atom_temp = self.atoms.copy()
        atom_temp.set_positions(x.reshape(-1, 3))
        calc = self.atoms.calc

        e = calc.get_property("energy", atoms=atom_temp)
        f = calc.get_property("forces", atoms=atom_temp)

        try:
            estd = calc.get_property("energy_std", atoms=atom_temp)
        except NotImplementedError:
            estd = 0.0
        try:
            fstd = calc.get_property("forces_std", atoms=atom_temp)
        except NotImplementedError:
            fstd = [0.0]*len(self.atoms)*3

        es = np.append(np.array(e).ravel(), np.array(f).ravel(), axis=0)
        fs = np.append(np.array(estd).ravel(), np.array(fstd).ravel(), axis=0)
        e_mean_f_mean_e_std_f_std = np.vstack((es, fs))
        return e_mean_f_mean_e_std_f_std

    def predict_mean_std_all(self, rs):
        """e_mean_f_mean_e_std_f_std.
        shape (n_sample, 2 [mean,std], n_atom+1 [e_ms,f_ms] )"""
        calc = self.atoms.calc
        if hasattr(calc, "calculate_batch"):
            atoms_list = []
            for x in rs:
                atom_temp = self.atoms.copy()
                atom_temp.set_positions(x.reshape(-1, 3))
                atoms_list.append(atom_temp)
            res = calc.calculate_batch(atoms_list)

            e = res["energy"]
            f = res["forces"]

            try:
                estd = res["energy_std"]
            except KeyError:
                estd = np.zeros_like(e)
            try:
                fstd = res["forces_std"]
            except KeyError:
                fstd = np.zeros_like(f)

            es = np.concatenate((np.array(e).reshape(-1, 1), np.array(f)), axis=1)
            fs = np.concatenate((np.array(estd).reshape(-1, 1), np.array(fstd)), axis=1)
            res = np.array([es, fs])
            res = res.transpose((1, 0, 2))
            return res

        else:
            res = [self.predict_mean_std(x) for x in rs]
            return np.array(res)

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

            self.update_batch(r_, e_, f_)

            random_factor = max(0.05 * 1/(self.nsteps+1), 0.005)  # decrease the degree of position change with step.

            r_ = self.resample_r(50, random_factor=random_factor)

            res = self.predict_mean_std_all(r_)

            ei = self.ei_balance(res)

            # ei = self.ei_balance(res, loss_coef=1-0.02*self.nsteps)
            # decrease the forces weight with step.
            # this should be just used for forces with
            index_ = np.argsort(-ei)[:10]
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



