# -*- coding: utf-8 -*-
import os
from typing import List

import numpy as np
import path
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError

from hpc2ml.data.batchdata import MtBatchData
from hpc2ml.data.structuretodata import StructureToData
from hpc2ml.nn.flow_geo import simple_predict
from hpc2ml.nn.metrics import mlm


class GNNEICalculater(Calculator):
    implemented_properties: List[str] = ["energy", "forces"]

    _deprecated = object()

    'Properties calculator can handle (energy, forces)'

    def __init__(self, model, resume_file, atoms=None, directory='.', convert=StructureToData(),
                 properties=None,
                 device="cpu",
                 **kwargs):
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory. Default
            is None: don't restart.

        directory: str or PurePath
            Working directory in which to read and write files and
            perform calculations.
        label: str
            Name used for all files.  Not supported by all calculators.
            May contain a directory, but please use the directory parameter
            for that instead.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        super(GNNEICalculater, self).__init__(atoms, directory=directory, **kwargs)

        self.model = model
        self.convert = convert
        self.convert.tq = False
        self.resume_file = resume_file
        self.device = torch.device(device)
        if properties is not None:
            self.implemented_properties = properties

        from hpc2ml.nn.flow_geo import load_check_point

        resume_file = path.Path(self.directory) / self.resume_file

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        model, optimizer, start_epoch, best_error, note = load_check_point(self.model,
                                                                           resume_file=resume_file,
                                                                           optimizer=optimizer,
                                                                           device=self.device)

        self.model = model
        self.optimizer = optimizer
        self.note = note
        self.start_epoch = start_epoch
        self.best_error = best_error

    def train(self, atoms, batch_data=None, **kwargs):
        """train"""
        print("Re-train model...")

        from torch_geometric.loader import DataLoader
        from hpc2ml.nn.flow_geo import LearningFlow

        if batch_data is None:
            dataset = self.get_bath_data(atoms, **kwargs)
        else:
            dataset = batch_data

        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        # # LearningFlow
        print_what = ""
        self.lf = LearningFlow(self.model, train_loader, test_loader=test_loader,
                               loss_method=mlm, optimizer=self.optimizer,
                               device=self.device, note=self.note, multi_loss=True,
                               target_name=tuple(self.implemented_properties),
                               checkpoint=True, store_filename=self.resume_file,
                               loss_threshold=0.001, print_what=print_what,
                               )

        self.lf.start_epoch = self.start_epoch
        self.lf.best_error = self.best_error
        self.lf.run(epoch=5)
        print("Done.")

    def get_bath_data(self, atoms=None, **kwargs) -> "MtBatchData":
        """Get scaled data."""
        batch = MtBatchData.from_atoms(atoms, **kwargs, convert=self.convert)
        batch.scale(dct=self.note)
        return batch

    def get_res_from_bath_data(self, batch: MtBatchData, msg_dict: dict) -> "MtBatchData":
        """Get unscaled data."""

        for k, v in msg_dict.items():
            ### energy and stress is atom
            slice_type = "sample"
            if k == "forces":
                slice_type = "atom"

            batch.add_prop(k, v, slice_type=slice_type)

        batch.unscale(dct=self.note)
        return batch

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError('{} property not implemented'
                                              .format(name))

        if atoms is None:
            atoms = self.atoms
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()
        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError('{} not present in this '
                                              'calculation'.format(name))

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def calculate(self, atoms=None, batch_data=None):
        assert isinstance(atoms, Atoms)
        properties = tuple(self.implemented_properties)

        if atoms is not None:
            self.atoms = atoms.copy()

        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

        from torch_geometric.loader import DataLoader

        if batch_data is None:
            batch = self.get_bath_data(atoms)
        else:
            batch = batch_data

        batch.to(self.device)

        predict_loader = DataLoader(batch, batch_size=32, shuffle=False)  # must not shuffle

        res = simple_predict(self.model, predict_loader, return_y_true=False, device=self.device,
                             process_out=None, process_label=None, target_name=properties,
                             multi_loss=True)

        res_dict = {k: v for k, v in zip(properties, res)}

        res_batch = self.get_res_from_bath_data(batch, res_dict)

        for k in properties:
            v = getattr(res_batch, k)

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()

            v = v.reshape(1, -1)

            if hasattr(v, "shape") and sum(v.shape) <= 2:
                v = v[0]

            self.results.update({k: v})

        if "stress" not in properties:
            self.results["stress"] = np.zeros((1, 6))
        return self.results

    def calculate_batch(self, atoms, batch_data=None):
        properties = tuple(self.implemented_properties)


        if isinstance(atoms, Atoms):
            atoms = [atoms, ]
        else:
            atoms = atoms

        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

        from torch_geometric.loader import DataLoader

        if batch_data is None:
            batch = self.get_bath_data(atoms)
        else:
            batch = batch_data

        l = len(atoms)

        batch.to(self.device)

        predict_loader = DataLoader(batch, batch_size=32, shuffle=False)

        res_batch = simple_predict(self.model, predict_loader, return_y_true=False,
                                   device=self.device, process_out=None,
                                   process_label=None, target_name=properties,
                                   multi_loss=True)

        res_dict = {k: v for k, v in zip(properties, res_batch)}

        res_batch = self.get_res_from_bath_data(batch, res_dict)

        res_list = []

        for i in range(l):
            results = {}
            res_batch_i = res_batch[i]
            for k in properties:
                v = getattr(res_batch_i, k)

                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()

                v = v.reshape(1, -1)

                results.update({k: v})

            if "stress" not in properties:
                results["stress"] = np.zeros((1, 6))
            res_list.append(results)
        return res_list
