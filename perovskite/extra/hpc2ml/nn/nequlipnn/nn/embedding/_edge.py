from typing import Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from hpc2ml.data.nequlip_support import AtomicDataDict
from hpc2ml.data.nequlip_support.AtomicDataDict import ST
from .._graph_mixin import GraphModuleMixin
from ..cutoffs import PolynomialCutoff
from ..radial_basis import BesselBasis


@compile_mode("script")
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps in_field
    """

    out_field: str

    def __init__(
            self,
            irreps_in=None,
            out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
            irreps_edge_sh: Union[int, str, o3.Irreps] = 2,
            edge_sh_normalization: str = "component",
            edge_sh_normalize: bool = True,

    ):
        super().__init__()
        self.out_field = out_field
        # irreps_edge_sh is the irreps_edge_sh

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: ST) -> ST:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh
        return data


@compile_mode("script")
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
            self,
            irreps_in=None,
            out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
            basis=BesselBasis, cutoff=PolynomialCutoff, basis_kwargs=None, cutoff_kwargs=None, ):
        super().__init__()

        if cutoff_kwargs is None:
            cutoff_kwargs = {}
        if basis_kwargs is None:
            basis_kwargs = {}

        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))])},
        )

    def forward(self, data: ST) -> ST:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_length_embedded = (
                self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        data[self.out_field] = edge_length_embedded
        return data
