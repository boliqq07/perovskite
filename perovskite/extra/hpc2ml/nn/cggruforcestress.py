import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, GRU
from torch_geometric.nn import MessagePassing, aggr
from torch_scatter import scatter

from hpc2ml.data.embedding.cohesive_energy import atomic_energy
from hpc2ml.function.support import add_edge_total
from hpc2ml.nn.activations import Act
from hpc2ml.nn.forces import Forces
from hpc2ml.nn.lj import LJ
from hpc2ml.nn.morse import Morse
from hpc2ml.nn.stress import Stress


class PotConv(MessagePassing):
    """Potential network.

    pot="morse" : Morse potential

    pot="lj": Lennard-Jones potential

    Keep edge_attr is the raw distance_vec, edge_weight is the raw care_ distance.
    """

    def __init__(self, in_channels, out_channels, nc_edge_hidden, pot="morse", mode="xyz", batch_norm=True):
        super().__init__(aggr='mean', flow="source_to_target")
        if pot == "morse":
            self.potlayer = Morse()
        elif pot == "lj":
            self.potlayer = LJ()
        else:
            raise NotImplementedError

        self.mode = mode

        if self.mode == "xyz":
            assert nc_edge_hidden == 3  # just for x,y,z vector.
        else:
            assert nc_edge_hidden == 1  # just for r vector.

        self.lin1 = nn.Linear(2 * in_channels + nc_edge_hidden + 1, 2 * in_channels, bias=False)
        self.lin1p = nn.Linear(2 * in_channels + 3, 2 * out_channels)  # not used perhaps
        self.lin2 = nn.Linear(2 * in_channels, 2 * out_channels)
        self.lin3 = nn.Linear(2 * out_channels, out_channels)

        self.act1_1p = Act("ssp")
        self.act2_3 = Act("ssp")

        self.batch_norm = batch_norm

        if batch_norm and self.mode == "xyz":
            self.bm = nn.BatchNorm1d(nc_edge_hidden)
        else:
            self.bm = nn.BatchNorm1d(1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.lin3.weight)

    def forward(self, x, edge_index, edge_weight, edge_attr, z):

        """keep edge_attr is the raw distance_vec, edge_weight is the raw care_ distance"""

        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr, z=z)
        x = self.lin2(x)
        x = self.act2_3(x)
        x = self.lin3(x)

        return x

    def message(self, z_i, z_j, x_i, x_j, edge_weight, edge_attr):
        """
        The input edge_index is the neighbor_index and center_index.

        1. If used edge_index (simple torch tensor object),
            Dut to the MessagePassing using flow='source_to_target' default. (i=1 and j=0)

            The edge_index_i=center_index,  edge_index_j=neighbor_index finally.

        2. If used adj_t (SparseTensor),

            In default,

            adj_t = SparseTensor(
                    row=store.edge_index[1], col=store.edge_index[0],
                    value=None if self.attr is None or self.attr not in store else
                    store[self.attr], sparse_sizes=store.size()[::-1],
                    is_sorted=True, trust_data=True)

            Finally, the edge_index_i=adj_t.storage.row()=center_index, edge_index_j=adj_t.storage.col()=neighbor_index,
            in MessagePassing.

        3.  If used edge_index (torch, and edge_index.is_sparse), (Not suggested !!!).
            edge_index_i = edge_index.indices()[0]
            edge_index_j = edge_index.indices()[1]
            please make sure the rank of center_index and neighbor_index in edge_index when you generate it!!!

        The rank is same in x (x_i and x_j),  z (x_i and x_j)
        """
        if self.mode == "xyz":
            return self.message_xyz(z_i, z_j, x_i, x_j, edge_weight, edge_attr)

        else:
            return self.message_r(z_i, z_j, x_i, x_j, edge_weight, edge_attr)

    def message_r(self, z_i, z_j, x_i, x_j, edge_weight, edge_attr):

        W = self.potlayer(cen=z_i, nei=z_j, r=edge_weight)
        if self.batch_norm:
            W = self.bm(W)
        x = self.lin1(torch.cat([x_i, x_j, edge_weight.view(-1, 1), edge_attr], dim=1))
        x = x * W
        return x

    def message_xyz(self, z_i, z_j, x_i, x_j, edge_weight, edge_attr):

        W = self.potlayer(cen=z_i, nei=z_j, r=edge_attr)
        if self.batch_norm:
            W = self.bm(W)
        x = self.lin1(torch.cat([x_i, x_j, edge_weight.view(-1, 1), edge_attr], dim=1))
        x = self.act1_1p(x)
        x = self.lin1p(torch.cat([x, W], dim=1))

        return x


class NNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nc_edge_hidden, cutoff=5.0, mlp=None):
        super().__init__(aggr='mean', flow="source_to_target")

        if mlp is None:
            self.mlp = nn.Sequential(nn.Linear(nc_edge_hidden, 32), ReLU(), nn.Linear(32, in_channels))
        else:
            self.mlp = mlp

        self.cutoff = cutoff

        self.lin1 = nn.Linear(in_channels, in_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels)
        self.lin3 = nn.Linear(out_channels, out_channels)
        self.act2_3 = Act("ssp")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.lin3.weight)
        self.lin2.bias.data.fill_(0)
        self.lin3.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, z=None):
        C = torch.cos(edge_weight * torch.pi / self.cutoff) * 0.5 + 0.5
        W = self.mlp(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        x = self.act2_3(x)
        x = self.lin3(x)

        return x

    def message(self, x_j, W):
        """
        The input edge_index is the neighbor_index and center_index.

        1. If used edge_index (simple torch tensor object),
            Dut to the MessagePassing using flow='source_to_target' default. (i=1 and j=0)

            The edge_index_i=center_index,  edge_index_j=neighbor_index finally.

        2. If used adj_t (SparseTensor),

            In default,

            adj_t = SparseTensor(
                    row=store.edge_index[1], col=store.edge_index[0],
                    value=None if self.attr is None or self.attr not in store else
                    store[self.attr], sparse_sizes=store.size()[::-1],
                    is_sorted=True, trust_data=True)

            Finally, the edge_index_i=adj_t.storage.row()=center_index, edge_index_j=adj_t.storage.col()=neighbor_index,
            in MessagePassing.

        3.  If used edge_index (torch, and edge_index.is_sparse), (Not suggested !!!).
            edge_index_i = edge_index.indices()[0]
            edge_index_j = edge_index.indices()[1]
            please make sure the rank of center_index and neighbor_index in edge_index when you generate it!!!

        The rank is same in x (x_i and x_j).
        """
        return x_j * W


class CGGRUForceStress(torch.nn.Module):
    """
    Note:
        1. direct_force=False and device="cuda:i" is un-compacted due to RNNNet can double backward.
        keep using device = "cpu" if direct_force=False."""

    def __init__(self, nfeat_node=19, nc_edge_hidden=1, dim=64, cutoff=5.0, n_block=3, get_force=True,
                 direct_force=True, get_stress=True, readout="set2set", mode="r", try_add_edge_msg=False,
                 use_pbc=True, pot="morse",
                 **kwargs):
        super().__init__()

        if direct_force is False and get_force is True:
            assert try_add_edge_msg == True  # Use to `_generate_graph` function.
            if nc_edge_hidden != 3:
                warnings.warn("nc_edge_hidden should changed to 3, due `_generate_graph` function just add edge_attr "
                              "with shape (n_bond,3)")
            nc_edge_hidden = 3  # Due to `_generate_graph` function just add edge_attr with shape (n_bond,3)

        self.lin0 = nn.Linear(nfeat_node, dim)
        self.act0 = Act("leaky_relu")

        if pot in ["morse", "lj"]:

            if nc_edge_hidden==1:
                mode ="r"
            elif nc_edge_hidden==3:
                mode ="xyz"
            else:
                raise NotImplementedError
            conv0 = PotConv(in_channels=dim, out_channels=dim, nc_edge_hidden=nc_edge_hidden, pot=pot, mode=mode)
        else:
            conv0 = NNConv(in_channels=dim, out_channels=dim, nc_edge_hidden=nc_edge_hidden, mlp=None, cutoff=cutoff)

        self.gru = GRU(dim, dim)

        assert n_block >= 2

        self.conv_list = nn.ModuleList([conv0, ])
        for i in range(n_block - 1):
            self.conv_list.append(NNConv(in_channels=dim, out_channels=dim,
                                         nc_edge_hidden=nc_edge_hidden,
                                         mlp=None, cutoff=cutoff))

        self.readout = readout

        if self.readout == "set2set":
            self.set2set = aggr.Set2Set(dim, processing_steps=3)
            self.lin1 = torch.nn.Linear(2 * dim, dim)
        else:
            self.lin1 = torch.nn.Linear(dim, dim)

        self.lin2 = torch.nn.Linear(dim, 1)
        self.readout = readout
        self.get_force = get_force
        self.get_stress = get_stress
        self.direct_force = direct_force

        self.cutoff = cutoff
        self.use_pbc = use_pbc
        self.try_add_edge_msg = try_add_edge_msg

        if self.direct_force:
            self.force_layer = Forces(dim)
        else:
            self.try_add_edge_msg = True

        if self.get_stress:
            self.stress_layer = Stress(input_dim=dim)

        energy_ref = torch.from_numpy(atomic_energy).view(-1, 1).float()
        self.register_buffer('energy_ref', energy_ref)

        self.atom_ref = nn.Embedding(100, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for interaction in self.conv_list:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    @staticmethod
    def _generate_graph(data, cutoff, use_pbc, max_neighbors=32):
        return add_edge_total(data, cutoff=cutoff, use_pbc=use_pbc, max_neighbors=max_neighbors,
                              use_exist_index=True)

    def forward(self, data):
        if not self.direct_force:
            data.pos.requires_grad_(True)

        if self.try_add_edge_msg:
            data = self._generate_graph(data, self.cutoff, self.use_pbc, max_neighbors=32)

        out = self.act0(self.lin0(data.x))
        h = out.unsqueeze(0)

        for convi in self.conv_list:
            m = F.leaky_relu(convi(out, data.edge_index, data.edge_weight, data.edge_attr, data.z.view(-1, 1)))
            if not self.direct_force:
                with torch.backends.cudnn.flags(enabled=False):
                    out, h = self.gru(m.unsqueeze(0), h)
            else:
                out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        if self.atom_ref is not None:
            out1 = out + self.atom_ref(data.z.view(-1))
        else:
            out1 = out

        if self.energy_ref is not None:
            out1 = out1 + self.energy_ref[data.z.view(-1)]

        if self.readout == "set2set":
            if not self.direct_force:
                with torch.backends.cudnn.flags(enabled=False):
                    energy = self.set2set(out1, data.batch)

            else:
                energy = self.set2set(out1, data.batch)

        else:
            energy = scatter(out1, data.batch, dim=0, reduce=self.readout)

        energy = F.leaky_relu(self.lin1(energy))
        energy = self.lin2(energy)
        energy = energy.view(-1)

        ########################

        if self.get_force:
            if not self.direct_force:
                forces = -1 * (
                    torch.autograd.grad(energy, data.pos,
                                        grad_outputs=torch.ones_like(energy),
                                        create_graph=True, )
                    [0])
            else:
                forces = self.force_layer(out)

            if self.get_stress:
                stress = self.stress_layer(out, data)

        if self.get_force:
            if self.get_stress:
                return energy, forces, stress
            else:
                return energy, forces
        else:
            return energy
