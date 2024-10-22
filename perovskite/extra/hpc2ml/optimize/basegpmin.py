# -*- coding: utf-8 -*-

# @Time  : 2023/2/2 11:22
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
"""This part is from ase.optimize and add more function."""

import warnings

import numpy as np
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.utils import write_json
from scipy import stats
from scipy.optimize import minimize

from hpc2ml.optimize.gp import GaussianProcess


class EISupport:
    # More
    def __init__(self, atoms=None, converged_std=None):
        self.atoms = atoms
        self.converged_std = converged_std
        self.force_consistent = False

    def _converged_force(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()
        return (forces ** 2).sum(axis=1).max() < self.fmax ** 2

    def _converged_energy_range(self, forces=None):
        """Did the optimization converge?"""
        _ = forces
        return len(self.es) < 3 or np.std(self.es) > 0.01

    def converged(self, forces=None):
        if self.converged_std == "forces":
            return self._converged_force(forces=forces)
        elif self.converged_std == "energy_range":
            return self._converged_energy_range(forces=forces)
        elif self.converged_std == "default":
            return self._converged(forces=forces)
        else:
            return self._converged_force(forces=forces)

    def _get_pos_r_e_f(self, atoms_list):

        e = []
        f = []
        r = []

        calc = atoms_list[0].calc

        if hasattr(calc, "calculate_batch"):
            res = calc.calculate_batch(atoms_list)
            r = np.array([i.positions.ravel() for i in atoms_list])
            e = res["energy"]
            f = res["forces"]
            index = np.isfinite(e)
            r = r[index]
            e = e[index]
            f = f[index]
        else:
            for atom_temp in atoms_list:

                ei = atom_temp.get_potential_energy(force_consistent=self.force_consistent)
                fi = atom_temp.get_forces().ravel()
                ri = atom_temp.positions.ravel()

                if np.isfinite(ei):
                    e.append(ei)
                    f.append(fi.reshape(1, -1))
                    r.append(ri.reshape(1, -1))

        if len(e) > 0:
            e = np.array(e).reshape(-1, 1)
            f = np.concatenate(f, axis=0) if not isinstance(f, np.ndarray) else f
            r = np.concatenate(r, axis=0) if not isinstance(r, np.ndarray) else r
        else:
            r = self.atoms.positions.reshape(1, -1)
            f = self.atoms.get_forces().reshape(1, -1)
            e = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            e = np.array(e).reshape(-1, 1)
        return r, e, f

    @staticmethod
    def _get_pos_r(atom_list):
        """and gather pos matrix from a batch of atoms """
        r = [i.positions.reshape(1, -1) for i in atom_list]
        r = np.concatenate(r, axis=0)
        return r

    def resample_atom(self, n, random_factor=0.001, axis_factor=(1, 1, 1)):
        """resample atoms."""
        # random_state = 0
        np.random.set_state(np.random.get_state())
        atoms_list = []
        for i in range(n):
            atom_temp = self.atoms.copy()
            atom_temp.calc = self.atoms.calc
            st_frac_coords = atom_temp.get_scaled_positions()
            axis_factor = np.array(list(axis_factor)).reshape(1, -1)
            st_frac_coords = np.array(st_frac_coords) + (
                    np.random.random(st_frac_coords.shape) - 0.5) * random_factor * axis_factor
            atom_temp.set_scaled_positions(st_frac_coords)
            atoms_list.append(atom_temp)

        return atoms_list

    def resample_r(self, n, random_factor=0.001, axis_factor=(1, 1, 1)):
        """resample atoms pos."""
        atom_list = self.resample_atom(n, random_factor=random_factor,
                                       axis_factor=axis_factor)
        r = self._get_pos_r(atom_list)
        return r

    def resample_r_e_f(self, n, random_factor=0.001, axis_factor=(1, 1, 1)):
        """resample atoms pos,energy,forces."""
        atom_list = self.resample_atom(n, random_factor=random_factor,
                                       axis_factor=axis_factor)
        r, e, f = self._get_pos_r_e_f(atom_list)
        return r, e, f

    def r_to_e_f(self, r):

        atoms_list = []
        for i in range(r.shape[0]):
            atom_temp = self.atoms.copy()
            atom_temp.calc = self.atoms.calc
            atom_temp.set_positions(r[i].reshape(-1, 3))
            atoms_list.append(atom_temp)

        r, e, f = self._get_pos_r_e_f(atoms_list)

        return r, e, f

    def r_to_e_f_single(self, ri, inplace=False):
        if not inplace:
            atom_temp = self.atoms.copy()
            atom_temp.calc = self.atoms.calc
        else:
            atom_temp = self.atoms

        atom_temp.set_positions(ri.reshape(-1, 3))
        ei = atom_temp.get_potential_energy(force_consistent=self.force_consistent)
        fi = atom_temp.get_forces().ravel()
        ri = ri.ravel()
        return ri, ei, fi

    def predict_mean_std(self, x):
        """This function should re-defined"""
        f, v = self.predict(x, get_variance=True) # predict is just for GP
        # e_mean = f[0]
        # f_mean = f[1:]
        v2 = np.diag(v)
        # e_std = v2[0]
        # f_std = v2[1:]
        e_mean_f_mean_e_std_f_std = np.vstack((f.ravel(), v2.ravel()))
        return e_mean_f_mean_e_std_f_std

    def predict_mean_std_all(self, xs):
        """e_mean_f_mean_e_std_f_std.
        shape (n_sample, 2 [mean,std], n_atom+1 [e_ms,f_ms] )"""
        res = [self.predict_mean_std(x) for x in xs]
        return np.array(res)

    @staticmethod
    def ei(mean_std, sign=1, target=None):

        mean = mean_std[:, 0]
        std = mean_std[:, 1]
        zero_std_index = (std < 0.001) & (std > -0.001)
        no_zero_std_index = ~zero_std_index

        # zero_std_index = std == 0.0
        # no_zero_std_index = std != 0.0
        ei = np.zeros_like(mean)

        if target is not None:

            upper = mean > target
            # lower = mean <= target

            mean[upper] = mean[upper] * -1  # trans value more than target to min problem

            mz = mean[zero_std_index]
            # sz = std[zero_std_index]

            if np.any(zero_std_index):
                ei0 = (mz - target) * 0.5  # 0.5 is stats.norm.cdf(0.0)
                ei[zero_std_index] = ei0

            mnz = mean[no_zero_std_index]
            snz = std[no_zero_std_index]

            if np.any(no_zero_std_index):
                kg = (mnz - target) / snz
                ei1 = mnz * stats.norm.cdf(kg) + snz * stats.norm.pdf(kg)
                ei[no_zero_std_index] = ei1

        else:
            mean = mean * sign  # trans min problem to max problem

            mz = mean[zero_std_index]
            # sz = std[zero_std_index]
            if np.any(zero_std_index):
                ei0 = (mz - np.max(mean)) * 0.5  # 0.5 is stats.norm.cdf(0.0)
                ei[zero_std_index] = ei0

            mnz = mean[no_zero_std_index]
            snz = std[no_zero_std_index]

            if np.any(no_zero_std_index):
                kg = (mnz - np.max(mean)) / snz
                ei1 = mnz * stats.norm.cdf(kg) + snz * stats.norm.pdf(kg)
                ei[no_zero_std_index] = ei1

        return ei

    def ei_balance(self, mean_std, loss_coef=1):

        fm = 0.0
        fn = mean_std.shape[2]

        ef_balance_coef = 1

        coef = loss_coef / (fn - 1) * ef_balance_coef

        ei_all = np.zeros((mean_std.shape[0]), dtype=np.float64)

        for i in range(fn):
            msi = mean_std[:, :, i]
            if i == 0:
                eii = self.ei(msi, sign=-1)  # less
            else:
                eii = self.ei(msi, target=fm)
                # more near to 0
            eii = eii.ravel()
            if np.max(eii) == np.min(eii):
                eii = 0.0
            else:
                eii = (eii - np.min(eii)) / (np.max(eii) - np.min(eii))
            if i > 0:
                eii = eii * coef
            ei_all += eii

        return ei_all.ravel()


class BaseGPMin(EISupport, Optimizer, GaussianProcess):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 prior=None, kernel=None, master=None, noise=None, weight=None,
                 scale=None, force_consistent=None, batch_size=None,
                 bounds=None, update_prior_strategy="maximum",
                 update_hyperparams=False, converged_std="forces"):

        # Warn the user if the number of atoms is very large
        if len(atoms) > 100:
            warning = ('Possible Memory Issue. There are more than '
                       '100 atoms in the unit cell. The memory '
                       'of the process will increase with the number '
                       'of steps, potentially causing a memory issue. '
                       'Consider using a different optimizer.')

            warnings.warn(warning)

        # Give it default hyperparameters
        if update_hyperparams:  # Updated GPMin
            if scale is None:
                scale = 0.3
            if noise is None:
                noise = 0.004
            if weight is None:
                weight = 2.
            if bounds is None:
                self.eps = 0.1
            elif bounds is False:
                self.eps = None
            else:
                self.eps = bounds
            if batch_size is None:
                self.nbatch = 1
            else:
                self.nbatch = batch_size
        else:  # GPMin without updates
            if scale is None:
                scale = 0.4
            if noise is None:
                noise = 0.001
            if weight is None:
                weight = 1.
            if bounds is not None:
                warning = ('The parameter bounds is of no use '
                           'if update_hyperparams is False. '
                           'The value provided by the user '
                           'is being ignored.')
                warnings.warn(warning, UserWarning)
            if batch_size is not None:
                warning = ('The parameter batch_size is of no use '
                           'if update_hyperparams is False. '
                           'The value provided by the user '
                           'is being ignored.')
                warnings.warn(warning, UserWarning)

            # Set the variables to something anyways
            self.eps = False
            self.nbatch = None

        self.strategy = update_prior_strategy
        self.update_hp = update_hyperparams
        self.function_calls = 1
        self.force_calls = 0
        self.x_list = []  # Training set features
        self.y_list = []  # Training set targets

        Optimizer.__init__(self, atoms, restart, logfile,
                           trajectory, master, force_consistent)
        if prior is None:
            self.update_prior = True
            prior = ConstantPrior(constant=None)
        else:
            self.update_prior = False

        if kernel is None:
            kernel = SquaredExponential()
        GaussianProcess.__init__(self, prior, kernel)
        self.set_hyperparams(np.array([weight, scale, noise]))
        EISupport.__init__(self, atoms=atoms, converged_std=converged_std)

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

        # Set/update the constant for the prior
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(self.y_list)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(self.y_list)[:, 0])
                self.prior.set_constant(max_e)

        # update hyperparams
        if (self.update_hp and self.function_calls % self.nbatch == 0 and
                self.function_calls != 0):
            self.fit_to_batch()

        # build the model
        self.train(np.array(self.x_list), np.array(self.y_list))

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

        # Set/update the constant for the prior
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(self.y_list)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(self.y_list)[:, 0])
                self.prior.set_constant(max_e)

        # update hyperparams
        if (self.update_hp and self.function_calls % self.nbatch == 0 and
                self.function_calls != 0):
            self.fit_to_batch()

        # build the model
        self.train(np.array(self.x_list), np.array(self.y_list))

    def remove(self, l):
        """should after update_batch"""

        self.x_list = self.x_list[:-l]
        self.y_list = self.y_list[:-l]

    def acquisition(self, r):
        e = self.predict(r)
        return e[0], e[1:]

    def fit_to_batch(self):
        """Fit hyperparameters keeping the ratio noise/weight fixed"""
        ratio = self.noise / self.kernel.weight
        self.fit_hyperparameters(np.array(self.x_list),
                                 np.array(self.y_list), eps=self.eps)
        self.noise = ratio * self.kernel.weight

    def relax_model(self, r0):
        result = minimize(self.acquisition, r0, method='L-BFGS-B', jac=True)
        if result.success:
            return result.x
        else:
            self.dump()
            warnings.warn("The minimization of the acquisition function "
                               "has not converged")
            return result.x

    def dump(self):
        """Save the training set"""
        if world.rank == 0 and self.restart is not None:
            with open(self.restart, 'wb') as fd:
                write_json(fd, (self.x_list, self.y_list))

    def read(self):
        self.x_list, self.y_list = self.load()

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
            r1 = self.relax_model(r0)

            self.atoms.set_positions(r1.reshape(-1, 3))
            e1 = self.atoms.get_potential_energy(force_consistent=fc)
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
