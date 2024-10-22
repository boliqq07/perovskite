"""Template model for crystal problem for learning and name unification.
Now, we don't suggest using this class."""
import warnings
from abc import abstractmethod

import ase.data as ase_data
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Embedding, Linear, LayerNorm, ModuleList, Softplus, ReLU, Sequential
from torch.nn import Module
from torch_scatter import segment_csr

from hpc2ml.nn.general import get_ptr


class TemplateCrystalModel(Module):
    """
    Template model for crystal problem for learning and name unification. Deprecation !

    The subclass should complete the `get_interactions_layer` and could redefined `get_readout_layer` function.

    Examples::
        def get_interactions_layer(self):
            self.layer_interaction = YourNet()

    Examples::
        def get_readout_layer(self):
            self.readout_layer = torch.nn.Sequential(...)

    Or re-write the `_forward` function。

    where the interactions are for center and neighbors interaction.
    where the readout_layer are for atomic properties merge into a whole.
    """

    def __init__(self,
                 nfeat_node=1,
                 nfeat_edge=3,
                 nfeat_state=0,

                 nc_node_hidden=16,
                 nc_edge_hidden=None,
                 nc_state_hidden=None,

                 edge_method=None,
                 node_method="z",
                 state_method=None,

                 nc_node_interaction=128,
                 num_interactions=1,

                 nnvocal=120,
                 nevocal=None,
                 nsvocal=None,

                 cutoff=10.0,
                 out_size=1,
                 readout='mean',

                 layer_interaction=None,
                 layer_readout=None,

                 add_node=None,
                 add_state=False,

                 norm=False,
                 mean=None,
                 std=None,
                 scale=None,
                 is_classification=False,
                 **kwargs
                 ):
        """
        Model for crystal problem.

        Args:
            nfeat_node: (int) input number of node feature (atom feature).
            nfeat_edge: (int) input number of bond feature. if ``num_edge_gaussians` offered,
            this parameter is neglect.
            nfeat_state: (int) input number of state feature.
            nnvocal: (int) number of atom, For generate the initial embedding matrix to on.
            nevocal: (int) number of bond types if bond attributes are types
            ngvocal: (int) number of global types if global attributes are types.
            behalf of node feature.
            nc_node_hidden: (int) nc_node_hidden for node feature.
            nc_node_interaction: (int) channels for node feature.
            num_interactions: (int) conv number.
            cutoff: (float) cutoff for calculate neighbor bond.
            readout: (str) Merge node method. such as "add","mean","max","mean".
            mean: (float) mean for y/energy.
            std: (float) std for y/energy.
            norm:(bool) False or True norm for y/energy.
            add_node: (torch.tensor shape (120,1)) properties for atom. such as target y is volumes of compound,
                add_node could be the atom volumes of all atom (H,H,He,Li,...). And you could copy the first term to
                make sure the `H` index start form 1.
            node_method: (bool,str) just used "z" or used "x" to calculate.
            layer_interaction: (Callable) torch module for interactions dynamically: pass the torch module to
            interactions parameter.static: re-define the ``get_interactions_layer`` and keep this parameter is None.
                the forward input is (h, edge_index, edge_weight, edge_attr, data=data)
            layer_readout: (Callable) torch module for interactions  dynamically: pass the torch module to
            interactions parameter. static: re-define the ``get_interactions_layer`` and keep this parameter is None.
            The forward input is (out,)
            add_state: (bool) add state attribute before output.
            is_classification: (bool) default False. 
            out_size:(int) number of out size. for regression,is 1 and for classification should be defined.
        """
        super(TemplateCrystalModel, self).__init__()

        warnings.warn("Template model for crystal problem for learning and name unification. "
                      "Now, we don't suggest using this class.", DeprecationWarning)

        self.interaction_kwargs = {}
        self.readout_kwargs = {}
        for k, v in kwargs.items():
            if "interaction_kwargs_" in k:
                self.interaction_kwargs[k.replace("interaction_kwargs_", "")] = v
            elif "readout_kwargs_" in k:
                self.readout_kwargs[k.replace("readout_kwargs_", "")] = v
            else:
                setattr(self, k, v)

        assert readout in ['add', 'sum', 'min', 'mean', "max"]

        if nnvocal < 120:
            print("Default, nvocal>=120, if you want simple the net work and "
                  "This network does not apply to other elements, the nvocal could be less but large than "
                  "the element type number in your data.")

        self.nnvocal = nnvocal
        self.nevocal = nevocal
        self.nsvocal = nsvocal

        self.nc_node_hidden = nc_node_hidden
        self.nc_state_hidden = nc_state_hidden
        self.nc_edge_hidden = nc_edge_hidden

        self.nfeat_state = nfeat_state
        self.nfeat_edge = nfeat_edge
        self.nfeat_node = nfeat_node

        self.node_method = node_method
        self.edge_method = edge_method
        self.state_method = state_method

        self.nc_node_interaction = nc_node_interaction
        self.num_interactions = num_interactions

        self.cutoff = cutoff
        self.readout = readout

        self.layer_interaction = layer_interaction
        self.layer_readout = layer_readout

        self.out_size = out_size

        self.norm = norm
        self.mean = mean
        self.std = std
        self.scale = scale
        self.add_state = add_state
        self.add_node = add_node
        self.is_classification = is_classification

        # 1. node (atom)
        if self.node_method == "z":
            if self.nfeat_node != 0:
                print("node_data=='z' would not use your self-defined 'x' data, "
                      "but element number Z.")
            self.layer_emb_node = Embedding(self.nnvocal, self.nc_node_hidden)
        elif self.node_method == "x":
            self.layer_emb_node = Sequential(Linear(self.nfeat_node, self.nc_node_hidden),
                                             Softplus(),
                                             Linear(self.nc_node_hidden, self.nc_node_hidden))
        else:
            raise ValueError("The node_data just accept 'z', 'x'.")

        # 2. edge
        if self.edge_method == "gaussian":
            assert self.nc_edge_hidden
            centers = kwargs.get("centers", None)
            self.layer_expand_edge = GaussianSmearing(0.0, self.cutoff, self.nc_edge_hidden)
        elif self.edge_method == "linear":
            assert self.nc_edge_hidden
            self.layer_expand_edge = Sequential(
                Linear(self.nfeat_edge, self.nc_edge_hidden),
                ReLU(),
                Linear(self.nc_edge_hidden, self.nc_edge_hidden), )
        elif self.edge_method == "embed":
            assert self.nevocal
            assert self.nc_edge_hidden
            self.layer_expand_edge = Sequential(
                Linear(self.nfeat_edge, self.nc_edge_hidden),
                ReLU(),
                Linear(self.nc_edge_hidden, self.nc_edge_hidden), )
        else:
            self.nc_edge_hidden = self.nfeat_edge

        # 3. state
        if self.nfeat_state is None and self.state_method == "embed":
            assert self.nevocal
            assert self.nc_state_hidden is not None
            # global state inputs are embedding integers
            self.layer_emb_state = Embedding(self.nsvocal, self.nc_state_hidden)

        else:
            self.nc_state_hidden = self.nfeat_state

        # 交互层 需要自定义 get_interactions_layer
        if layer_interaction is None:
            self.get_interactions_layer()
        elif isinstance(layer_interaction, ModuleList):
            self.SimpleGeoResNet(layer_interaction)
        elif isinstance(layer_interaction, Module):
            self.layer_interaction = layer_interaction
        else:
            raise NotImplementedError("Please implement get_interactions_layer function, "
                                      "or pass layer_interaction parameters.")
        # 合并层 需要自定义
        if layer_readout is None:
            self.get_readout_layer()
        elif isinstance(layer_readout, Module):
            self.layer_readout = layer_readout
        else:
            raise NotImplementedError("please implement get_readout_layer function, "
                                      "or pass layer_readout parameters.")

        # 原子性质嵌入
        if add_node is not None:
            self.add_node_layer = EmbAtomsProp(add_node)

        if self.add_state:
            assert self.nfeat_state > 0
            self.add_state_layer = Sequential(
                LayerNorm(self.nfeat_state),
                ReLU(),
                Linear(self.nfeat_state, 2 * self.nfeat_state),
                ReLU(),
                Linear(2 * self.nfeat_state, 2 * self.nfeat_state),
                ReLU(),
                Linear(2 * self.nfeat_state, self.out_size),
            )

        self.reset_parameters()

    def _forward(self, h, data):
        """This function can be re-write."""

        h = self.layer_interaction(h, data=data)

        if self.add_node is not None:
            h = h + self.add_node_layer(data.z)

        h = self.layer_readout(h, data)

        if self.add_state:
            assert hasattr(data, "state_attr"), "the ``add_state`` must accept ``state_attr`` in data."
            sta = self.add_state_layer(data.state_attr)
            h = h + sta

        return h

    def forward(self, data):

        assert hasattr(data, "z")
        assert hasattr(data, "pos")
        assert hasattr(data, "batch")
        z = data.z

        # 原子处理数据阶段
        if self.node_method == "z":
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.layer_emb_node(z)
        elif self.node_method == "x":
            assert hasattr(data, "x")
            h = self.layer_emb_node(data.x)
        else:
            raise NotImplementedError

        data.x = h

        # 键处理数据阶段
        if self.nc_edge_hidden:
            if self.edge_method == "linear":
                edge_attr = self.layer_expand_edge(data.edge_attr)
                data.edge_attr = edge_attr
            elif self.edge_method == "embed":
                assert data.edge_weight.dtype == torch.int64
                edge_attr = self.layer_expand_edge(data.edge_weight)
                data.edge_attr = edge_attr
            elif self.edge_method == "gaussian":
                edge_attr = self.layer_expand_edge(data.edge_weight)
                data.edge_attr = edge_attr
            else:
                pass

        if hasattr(data, "state_attr"):
            if self.nfeat_state is None and self.state_method == "embed":
                assert data.edge_weight.dtype == torch.int64
                data.edge_attr = self.layer_emb_state(data.edge_attr)
            else:
                pass

        h = self._forward(h, data)  # key function.

        h = self.output_forward(h)

        return h

    @abstractmethod
    def get_interactions_layer(self):
        """This part shloud re-defined. And must add the ``interactions`` attribute.

        Examples::

            def get_interactions_layer(self):
                self.layer_interaction = YourNet()

        """

    def get_readout_layer(self):
        """This part should re-defined. And must add the ``readout_layer`` attribute.

        Examples::
            def get_readout_layer(self):
                self.readout_layer = torch.nn.Sequential(...)

        Examples::

            def get_readout_layer(self):
                self.readout_layer = YourNet()

        """
        if "readout_kwargs_layers_size" in self.readout_kwargs:
            self.layer_readout = GeneralReadOutLayer(**self.readout_kwargs)
        else:
            if self.is_classification:
                last_layer = nn.Sigmoid
            else:
                last_layer = None
            self.layer_readout = GeneralReadOutLayer(
                [self.nc_node_interaction, self.readout, 2 * self.nc_node_interaction,
                 2 * self.nc_node_interaction, self.out_size], last_layer=last_layer, **self.readout_kwargs)

    def output_forward(self, out):
        """Last dispose."""
        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean
        if self.norm is True:
            out = torch.norm(out, dim=-1, keepdim=True)
        if self.scale is not None:
            out = self.scale * out
        return out

    def forward_weight_attr(self, data):
        """Add edge_weight or edge_attr by another, use mannully."""

        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            if not hasattr(data, "edge_attr") or data.edge_attr is None:
                raise NotImplementedError("Must offer edge_weight or edge_attr.")
            else:
                if data.edge_attr.shape[1] == 1:
                    data.edge_weight = data.edge_attr.reshape(-1)
                else:
                    data.edge_weight = torch.norm(data.edge_attr, dim=1, keepdim=True)

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            if not hasattr(data, "edge_weight") or data.edge_weight is None:
                raise NotImplementedError("Must offer edge_weight or edge_attr.")
            else:
                data.edge_attr = data.edge_weight.reshape(-1, 1)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'nc_node_hidden={self.nc_node_hidden}, '
                f'nc_edge_hidden={self.nc_edge_hidden}, '
                f'nc_state_hidden={self.nc_state_hidden}, '
                f'nc_node_interaction={self.nc_node_interaction}, '
                f'num_interactions={self.num_interactions}, '
                f'cutoff={self.cutoff})')

    def reset_parameters(self):
        if hasattr(self, "layer_emb_node"):
            try:
                self.layer_emb_node.reset_parameters()
            except AttributeError:
                pass
        if hasattr(self, "layer_expand_edge"):
            try:
                self.layer_expand_edge.reset_parameters()
            except AttributeError:
                pass
        if hasattr(self, "layer_emb_state"):
            try:
                self.layer_emb_state.reset_parameters()
            except AttributeError:
                pass
        if hasattr(self, "layer_interaction"):
            try:
                self.layer_interaction.reset_parameters()
            except AttributeError:
                pass
        if hasattr(self, "layer_readout"):
            try:
                self.layer_readout.reset_parameters()
            except AttributeError:
                pass


class _ReadOutLayer(Module):
    """Merge node."""

    def __init__(self, channels, out_size=1, readout="add"):
        super(_ReadOutLayer, self).__init__()
        self.readout = readout
        self.ro_lin1 = Linear(channels, channels * 10)
        self.ro_lrelu1 = nn.LeakyReLU()
        self.ro_lin2 = Linear(channels * 10, channels * 5)
        self.ro_lrelu2 = nn.LeakyReLU()
        self.ro_lin3 = Linear(channels * 5, out_size)

    def forward(self, h, data):
        batch = data.batch
        h = self.ro_lin1(h)
        h = self.ro_lrelu1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        h = self.ro_lin2(h)
        h = self.ro_lrelu2(h)
        h = self.ro_lin3(h)
        return h


class EmbAtomsProp(Module):
    """Embedding of atomic properties."""

    def __init__(self, array="atomic_radii"):

        super().__init__()

        if array == "atomic_mass":
            array = torch.from_numpy(ase_data.atomic_masses)  # 嵌入原子质量
        elif array == "atomic_radii":
            array = torch.from_numpy(ase_data.covalent_radii)  # 嵌入共价半径
        elif isinstance(array, np.ndarray):
            assert array.shape[0] == 120
            array = torch.from_numpy(array)
        elif isinstance(array, Tensor):
            assert array.shape[0] == 120
        else:
            raise NotImplementedError("just accept str,np,ndarray or tensor with shape (120,)")
        # 嵌入原子属性，需要的时候运行本函数
        # (嵌入别太多，容易慢，大多数情况下用不到。)
        # 缓冲buffer必须要登记注册才会有效,如果仅仅将张量赋值给Module模块的属性,不会被自动转为缓冲buffer.
        # 因而也无法被state_dict()、buffers()、named_buffers()访问到。
        self.register_buffer('atomic_temp', array)

    def forward(self, z):
        return self.atomic_temp[z]


class GeneralReadOutLayer(Module):
    """General Merge node.
    """

    def __init__(self, layers_size=(128, "add", 32, 1), last_layer=None, active_layer_type="Softplus"):
        super(GeneralReadOutLayer, self).__init__()
        l = len(layers_size)
        readout = [i for i in layers_size if isinstance(i, str)]
        if active_layer_type == "Softplus":
            active_layer = nn.Softplus
        elif active_layer_type == "SoftplusShift":
            active_layer = ShiftedSoftplus
        elif active_layer_type == "ReLU":
            active_layer = nn.ReLU
        elif active_layer_type == "LeakyReLU":
            active_layer = nn.LeakyReLU
        elif isinstance(active_layer_type, Module):
            active_layer = active_layer_type.__class__
        else:
            raise NotImplementedError("Can't identify the type of layer.")

        assert len(readout) == 1, "The readout layer must be set one time, please there are {} layer: {}.".format(
            len(readout), readout)

        readout_site = [n for n, i in enumerate(layers_size) if isinstance(i, str)][0]
        readout = layers_size[readout_site]
        assert readout in ('sum', 'max', 'min', 'mean', 'add')
        self.readout = readout
        layers_size = list(layers_size)
        layers_size[readout_site] = layers_size[readout_site - 1]

        part_first = []
        part_second = []
        i = 0
        while i < l - 1:
            if i < readout_site - 1:
                part_first.append(Linear(layers_size[i], layers_size[i + 1]))
                part_first.append(active_layer())
            elif i < readout_site:
                pass
            else:
                part_second.append(active_layer())
                part_second.append(Linear(layers_size[i], layers_size[i + 1]))

            i += 1

        self.ro_seq1 = Sequential(*part_first)
        self.ro_seq2 = Sequential(*part_second)
        self.ro_last_layer = last_layer

    def reset_parameters(self):
        self.ro_seq1.reset_parameters()
        self.ro_seq2.reset_parameters()
        self.ro_last_layer.reset_parameters()

    def forward(self, h, data):
        batch = data.batch
        if len(self.ro_seq1) > 0:
            h = self.ro_seq1(h)
        h = segment_csr(h, get_ptr(batch), reduce=self.readout)
        if len(self.ro_seq2) > 0:
            h = self.ro_seq2(h)
        if self.ro_last_layer is not None:
            h = self.ro_last_layer(h)
        return h


class SimpleGeoResNet(Module):
    """Simple ResNet."""

    def __init__(self, module_list: ModuleList):
        super().__init__()
        self.res_list = module_list

    def forward(self, h, data):
        h0 = h
        for interaction in self.res_list:
            h = interaction(h0, data=data)
        return h


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussian=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussian)
        self.coef = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coef * torch.pow(dist, 2))


def get_active_layer(name: str):
    """Return active layer."""
    if name == "Softplus":
        active_layer = nn.Softplus
    elif name == "SoftplusShift":
        active_layer = ShiftedSoftplus
    elif name == "ReLU":
        active_layer = nn.ReLU
    elif name == "LeakyReLU":
        active_layer = nn.LeakyReLU
    else:
        raise NotImplementedError
    return active_layer
