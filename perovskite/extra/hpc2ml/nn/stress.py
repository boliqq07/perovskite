import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import aggr
from torch_scatter import scatter


class Stress(nn.Module):
    """Stress net"""

    def __init__(self, input_dim=3, readout="mean"):
        super(Stress, self).__init__()

        self.mlp0 = nn.Sequential(nn.Linear(input_dim + 3 + 81, 128, bias=True), nn.ReLU())

        self.bm = nn.BatchNorm1d(input_dim + 3 + 81)

        self.mlp2 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64))

        self.readout = readout

        if self.readout == "set2set":
            self.set2set = aggr.Set2Set(64, processing_steps=2)

            self.mlp3 = nn.Sequential(
                nn.Linear(64 * 2, 32), nn.ReLU(),
                nn.Linear(32, 6))
        else:
            self.mlp3 = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 6))

    def forward(self, atom_prop, data):
        """
        # atom_prop size (n_atoms, 3)
        # pos (n_atoms, 3)
        # cell (n_sturcture, 3, 3)
        """

        pos = data.pos
        cell = data.cell
        cell_ravel = cell.view(-1, 9)
        cell_ravel = (cell_ravel.unsqueeze(1) * cell_ravel.unsqueeze(-1)).view(-1, 81)
        cell_ravel = cell_ravel[data.batch]

        if not hasattr(data, "frac_pos"):
            cell_1 = torch.inverse(cell)
            frac_pos = torch.bmm(pos.unsqueeze(1), cell_1[data.batch], ).squeeze(1)
        else:
            frac_pos = data.frac_pos

        h = torch.cat((atom_prop, frac_pos, cell_ravel), dim=1)

        h = self.bm(h)
        h = self.mlp0(h)
        h = self.mlp2(h)

        if self.readout == "set2set":
            h = self.set2set(h, data.batch)
            stress = self.mlp3(h)
        else:
            h = scatter(h, data.batch, dim=0, reduce=self.readout)
            stress = self.mlp3(h)

        return stress


# class StressOutput(torch.nn.Module):
#     r"""Compute stress (and forces) using autograd of an energy model.
#
#     See:
#         Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
#         https://pure.mpg.de/rest/items/item_2085135_9/component/file_2156800/content
#
#     Args:
#         func: the energy model to wrap
#     """
#
#     def __init__(self, func):
#         super().__init__()
#
#         self.do_forces = True
#
#         self.func = func
#
#     def forward(self, data: Data):
#         batch = data.batch
#         num_batch = int(batch.max().cpu().item()) + 1
#         pos = data.pos
#
#         orig_cell = data.cell
#         # Make the cell per-batch
#         cell = orig_cell.view(-1, 3, 3).expand(num_batch, 3, 3)
#
#         displacement = torch.zeros(
#             (num_batch, 3, 3),
#             dtype=pos.dtype,
#             device=pos.device,
#         )
#         displacement.requires_grad_(True)
#         data["_displacement"] = displacement
#         # in the above paper, the infinitesimal distortion is *symmetric*
#         # so we symmetrize the displacement before applying it to
#         # the positions/cell
#         # This is not strictly necessary (reasoning thanks to Mario):
#         # the displacement's asymmetric 1o term corresponds to an
#         # infinitesimal rotation, which should not affect the final
#         # output (invariance).
#         # That said, due to numerical error, this will never be
#         # exactly true. So, we symmetrize the deformation to
#         # take advantage of this understanding and not rely on
#         # the invariance here:
#         symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
#         did_pos_req_grad: bool = pos.requires_grad
#         pos.requires_grad_(True)
#         # bmm is natom in batch
#         data["pos"] = pos + torch.bmm(
#             pos.unsqueeze(-2), symmetric_displacement[batch]
#         ).squeeze(-2)
#         # we only displace the cell if we have one:
#
#         data["cell"] = cell + torch.bmm(
#             cell, symmetric_displacement
#         )
#
#         # Call model and get gradients
#         energy = self.func(data)
#
#         grads = torch.autograd.grad(
#             [energy.sum()],
#             [pos, data["_displacement"]],
#             create_graph=self.training,  # needed to allow gradients of this output during training
#         )
#
#         # Put negative sign on forces
#         forces = grads[0]
#         forces = torch.neg(forces)
#
#         # data['forces'] = forces
#
#         # Store virial
#         virial = grads[1]
#
#         volume = torch.einsum(
#             "zi,zi->z",
#             cell[:, 0, :],
#             torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
#         ).unsqueeze(-1)
#         stress = virial / volume.view(-1, 1, 1)
#
#         data["cell"] = orig_cell
#
#         # data["stress"] = stress
#
#         virial = torch.neg(virial)
#
#         # data["virial"] = virial
#
#         # Remove helper
#         del data["_displacement"]
#         if not did_pos_req_grad:
#             # don't give later modules one that does
#             pos.requires_grad_(False)
#
#         return energy, forces, stress, virial

# class Stress(nn.Module):
#     """Stress net"""
#     def __init__(self, input_dim=3):
#         super(Stress, self).__init__()
#
#         self.mlp0 = nn.Linear(input_dim, 32)
#
#         self.mlp1 = nn.Linear(3, 3, bias=True)
#         self.mlp1n = nn.Linear(3, 3, bias=True)
#
#         self.mlp2 = nn.Sequential(nn.Linear(87+32, 32), nn.ReLU(),
#                                   nn.Linear(32, 16), nn.ReLU(),
#                                   nn.Linear(16, 6), )
#         self.reset_parameters()
#         self.readout = "set2set"
#         self.readout = "mean"
#
#         if self.readout == "set2set":
#             self.set2set = aggr.Set2Set(6, processing_steps=2)
#             self.lin1 = torch.nn.Linear(2 * 6, 6)
#         else:
#             self.lin1 = torch.nn.Linear(6, 6)
#
#     def reset_parameters(self) -> None:
#         nn.init.eye_(self.mlp1.weight)
#         nn.init.eye_(self.mlp1n.weight)
#         self.mlp1.bias.data.fill_(0)
#         self.mlp1n.bias.data.fill_(0)
#
#         for m in self.mlp2:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0)
#
#     def forward(self, atom_prop, data):
#         """
#         # atom_prop size (n_atoms, 3)
#         # pos (n_atoms, 3)
#         # cell (n_sturcture, 3, 3)
#         """
#
#         # atom_prop = torch.rand_like(data.pos) # for test
#
#         pos = data.pos
#         cell = data.cell
#         cell_ravel = cell.view(-1, 9)
#         cell_ravel = (cell_ravel.unsqueeze(1) * cell_ravel.unsqueeze(-1)).view(-1, 81)
#         cell_ravel = cell_ravel[data.batch]
#
#         if not hasattr(data,"frac_pos"):
#             cell_1 = torch.inverse(cell)
#             frac_pos = torch.bmm(pos.unsqueeze(1), cell_1[data.batch], ).squeeze(1)
#         else:
#             frac_pos = data.frac_pos
#
#         threshold = frac_pos - torch.floor(frac_pos) - 0.5
#
#         # threshold.requires_grad=False
#
#         atom_prop = self.mlp0(atom_prop)
#
#         threshold1 = F.relu(self.mlp1(threshold))
#         threshold2 = F.relu(self.mlp1n(-threshold))
#
#         h = torch.cat((atom_prop, threshold1, threshold2, cell_ravel), dim=1)
#
#         h = self.mlp2(h)
#
#         if self.readout == "set2set":
#             h = F.relu(self.set2set(h, data.batch))
#
#         else:
#             h = F.relu(scatter(h, data.batch, dim=0, reduce=self.readout))
#
#         stress = self.lin1(h)
#
#         return stress
