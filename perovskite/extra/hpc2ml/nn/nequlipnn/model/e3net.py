from typing import Union

import torch
from e3nn import o3
from torch_geometric.data import Batch, Data

from hpc2ml.data.nequlip_support import AtomicDataDict
from hpc2ml.data.batchdata import to_atomicdatadict
from hpc2ml.nn.nequlipnn.nn import AtomwiseLinear, ConvNetLayer, InteractionBlock, AtomwiseReduce
from hpc2ml.nn.nequlipnn.nn.cutoffs import PolynomialCutoff
from hpc2ml.nn.nequlipnn.nn.embedding import OneHotAtomEncoding, SphericalHarmonicEdgeAttrs, \
    RadialBasisEdgeEncoding
from hpc2ml.nn.nequlipnn.nn.radial_basis import BesselBasis
from nequip.nn import GraphModuleMixin


def config(lmax=1, parity=True, num_features=16, **kw):
    kw_ = {"readout": "sum",
           "r_max": 4.0,
           "num_layers": 3,
           "num_basis": 8,
           "num_types": 80,
           "PolynomialCutoff_p": 6,
           "nonlinearity_type": "gate"}

    kw.update({"chemical_embedding_irreps_out": repr(o3.Irreps([(num_features, (0, 1))]))})  # n scalars

    kw.update({"irreps_edge_sh": repr(o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1))})

    kw.update({"feature_irreps_hidden": repr(o3.Irreps(
        [(num_features, (l, p))
         for p in ((1, -1) if parity else (1,))
         for l in range(lmax + 1)
         ]))})

    kw.update({"conv_to_output_hidden_irreps_out": repr(o3.Irreps([(max(1, num_features // 2), (0, 1))]))})
    kw_.update(kw)
    return kw_


class NeqlipNN(GraphModuleMixin, torch.nn.Module):
    def __init__(self, lmax=1, parity=True, num_features=32,
                 ):
        kw = config(lmax=lmax, parity=parity, num_features=num_features, )
        num_types = kw["num_types"]
        irreps_edge_sh = kw["irreps_edge_sh"]
        num_basis = kw["num_basis"]
        r_max = kw["r_max"]
        PolynomialCutoff_p = kw["PolynomialCutoff_p"]
        feature_irreps_hidden = kw["feature_irreps_hidden"]
        num_layers = kw["num_layers"]
        nonlinearity_type = kw["nonlinearity_type"]
        readout = kw["readout"]
        conv_to_output_hidden_irreps_out = kw["conv_to_output_hidden_irreps_out"]

        super(NeqlipNN, self).__init__()

        self.try_add_edge = True

        # default node_attrs num_types x 0e
        self.one_hot = OneHotAtomEncoding(
            irreps_in=None,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
            num_types=num_types,
            copy_features=True,
        )

        irreps_temp = self.one_hot.irreps_out

        # default edge_attrs out = irreps_edge_sh=o3.Irreps.spherical_harmonics(3, p=-1),
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_in=irreps_temp,
                                                       out_field=AtomicDataDict.EDGE_ATTRS_KEY,
                                                       irreps_edge_sh=irreps_edge_sh,
                                                       edge_sh_normalization="component",
                                                       edge_sh_normalize=True)

        irreps_temp = self.spharm_edges.irreps_out

        # default edge_embedding num_basis x 0e
        self.radial_basis = RadialBasisEdgeEncoding(irreps_in=irreps_temp,
                                                    out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                                                    basis=BesselBasis, cutoff=PolynomialCutoff,
                                                    basis_kwargs={"r_max": r_max, "num_basis": num_basis},
                                                    cutoff_kwargs={"r_max": r_max, "p": PolynomialCutoff_p},
                                                    )

        irreps_temp = self.radial_basis.irreps_out

        self.chemical_embedding = AtomwiseLinear(in_field=AtomicDataDict.NODE_FEATURES_KEY,
                                                 out_field=AtomicDataDict.NODE_FEATURES_KEY,
                                                 irreps_in=irreps_temp,
                                                 irreps_out=feature_irreps_hidden)

        irreps_temp = self.chemical_embedding.irreps_out

        for i in range(num_layers):
            ncl = ConvNetLayer(
                irreps_in=irreps_temp, feature_irreps_hidden=feature_irreps_hidden,
                convolution=InteractionBlock, nonlinearity_type=nonlinearity_type,
                resnet=False,
                convolution_kwargs={
                    "irreps_in": irreps_temp,
                    "irreps_out": irreps_temp,
                    "invariant_layers": 2,
                    "invariant_neurons": 32,
                    "avg_num_neighbors": None,
                    "use_sc": True,
                    "nonlinearity_scalars": {"e": "silu"}, }, )
            setattr(self, f"layer{i}_convnet", ncl)

        irreps_temp = self.chemical_embedding.irreps_out

        self.conv_to_output_hidden = AtomwiseLinear(in_field=AtomicDataDict.NODE_FEATURES_KEY,
                                                    out_field=AtomicDataDict.NODE_FEATURES_KEY,
                                                    irreps_in=irreps_temp,
                                                    irreps_out=conv_to_output_hidden_irreps_out)  # don't change size

        irreps_temp = self.conv_to_output_hidden.irreps_out

        self.output_hidden_to_scalar = AtomwiseLinear(in_field=AtomicDataDict.NODE_FEATURES_KEY,
                                                      out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                                                      irreps_in=irreps_temp, irreps_out="1x0e")  # scalar

        irreps_temp = self.output_hidden_to_scalar.irreps_out
        self.output_reduce = AtomwiseReduce(in_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                                            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
                                            reduce=readout,
                                            irreps_in=None, )


        self._init_irreps(
            irreps_in={},
            irreps_out=irreps_temp.copy(),
        )

    def forward(self, data: Union[Batch, Data, dict]):

        if not isinstance(data, dict):
            data = to_atomicdatadict(data)

        if self.try_add_edge:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        for module in self.children():
            data = module(data)

        data[AtomicDataDict.TOTAL_ENERGY_KEY] = data[AtomicDataDict.TOTAL_ENERGY_KEY].view(-1)
        return data
