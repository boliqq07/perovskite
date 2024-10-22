# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# @Time  : 2022/9/27 15:12
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import torch
from torch import nn
from torch.nn import ReLU


#     missing,  # X
#     0.31,  # H
#     0.28,  # He
#     1.28,  # Li
#     0.96,  # Be
#     0.84,  # B
#     0.76,  # C
#     0.71,  # N
#     0.66,  # O
#     0.57,  # F
#     0.58,  # Ne
#     1.66,  # Na
#     1.41,  # Mg
#     1.21,  # Al
#     1.11,  # Si
#     1.07,  # P
#     1.05,  # S
#     1.02,  # Cl
#     1.06,  # Ar
#     2.03,  # K
#     1.76,  # Ca
#     1.70,  # Sc
#     1.60,  # Ti
#     1.53,  # V
#     1.39,  # Cr
#     1.39,  # Mn
#     1.32,  # Fe
#     1.26,  # Co
#     1.24,  # Ni
#     1.32,  # Cu
#     1.22,  # Zn
#     1.22,  # Ga
#     1.20,  # Ge
#     1.19,  # As
#     1.20,  # Se
#     1.20,  # Br
#     1.16,  # Kr
#     2.20,  # Rb
#     1.95,  # Sr
#     1.90,  # Y
#     1.75,  # Zr
#     1.64,  # Nb
#     1.54,  # Mo
#     1.47,  # Tc
#     1.46,  # Ru
#     1.42,  # Rh
#     1.39,  # Pd
#     1.45,  # Ag
#     1.44,  # Cd
#     1.42,  # In
#     1.39,  # Sn
#     1.39,  # Sb
#     1.38,  # Te
#     1.39,  # I
#     1.40,  # Xe
#     2.44,  # Cs
#     2.15,  # Ba
#     2.07,  # La
#     2.04,  # Ce
#     2.03,  # Pr
#     2.01,  # Nd
#     1.99,  # Pm
#     1.98,  # Sm
#     1.98,  # Eu
#     1.96,  # Gd
#     1.94,  # Tb
#     1.92,  # Dy
#     1.92,  # Ho
#     1.89,  # Er
#     1.90,  # Tm
#     1.87,  # Yb
#     1.87,  # Lu
#     1.75,  # Hf
#     1.70,  # Ta
#     1.62,  # W
#     1.51,  # Re
#     1.44,  # Os
#     1.41,  # Ir
#     1.36,  # Pt
#     1.36,  # Au
#     1.32,  # Hg
#     1.45,  # Tl
#     1.46,  # Pb
#     1.48,  # Bi
#     1.40,  # Po
#     1.50,  # At
#     1.50,  # Rn
#     2.60,  # Fr
#     2.21,  # Ra
#     2.15,  # Ac
#     2.06,  # Th
#     2.00,  # Pa
#     1.96,  # U
#     1.90,  # Np
#     1.87,  # Pu
#     1.80,  # Am
#     1.69,  # Cm
#     missing,  # Bk
#     missing,  # Cf
#     missing,  # Es
#     missing,  # Fm
#     missing,  # Md
#     missing,  # No
#     missing,  # Lr
#     missing,  # Rf
#     missing,  # Db
#     missing,  # Sg
#     missing,  # Bh
#     missing,  # Hs
#     missing,  # Mt
#     missing,  # Ds
#     missing,  # Rg
#     missing,  # Cn
#     missing,  # Nh
#     missing,  # Fl
#     missing,  # Mc
#     missing,  # Lv
#     missing,  # Ts
#     missing,  # Og


class LJ(torch.nn.Module):
    """LJ potential"""

    def __init__(self, cutoff=5.0):
        super().__init__()
        from ase.data import covalent_radii
        radii = torch.from_numpy(covalent_radii).float()
        self.lin1 = nn.Sequential(
            nn.Linear(2, 6, bias=True), ReLU(),
            nn.Linear(6, 6, bias=True), ReLU(),
            nn.Linear(6, 6, bias=True)
        )
        self.cutoff = cutoff  # ? old
        self.register_buffer('radii', radii)

    def forward(self, cen, nei, r):
        cen = cen.view(-1)
        nei = nei.view(-1)
        if r.ndim == 1 or (r.ndim == 2 and r.shape[1] == 1):
            r = r.view(-1, 1)
            return self._forward_r(cen, nei, r)
        elif r.ndim == 2 and r.shape[1] == 3:
            return self._forward_xyz(cen, nei, r)
        else:
            raise NotImplementedError("r are just accept distance (N,) or (N,1) or distance_vec (N,3)")

    def _forward_r(self, nei, cen, r):
        r = r.view(-1, 1)
        r0 = (self.radii[nei] + self.radii[nei]).view(-1, 1)

        param_in = torch.vstack((nei + cen, nei * cen)).T.float()
        param_out = self.lin1(param_in)
        param_out = torch.abs(param_out)

        epsilon, k, c, n_3, n1, n2, = torch.chunk(param_out, chunks=6, dim=1)

        n1 = 0.001 * n1
        n2 = 0.001 * n2
        n_3 = 0.001 * n_3

        MV_COEF = 0.56
        LINEBASE_COEF = 0.5

        sigma = LINEBASE_COEF * (0.8 + 0.01 * k) * r0 / MV_COEF

        sigma = sigma / r

        pot = 4 * epsilon * (torch.pow(sigma, 12) - torch.pow(sigma, 6)) + n_3 * r ** (1 / 3) + n2 * r ** 2

        pot = torch.minimum(pot, torch.full_like(pot, 10))
        pot = pot - n1 * r + c

        return pot

    def _forward_xyz(self, cen, nei, r):

        pot = self._forward_r(cen, nei, torch.sum(r ** 2, dim=1, keepdim=True))

        r2 = r ** 2

        pot = pot * r2 / torch.sum(r2, dim=1, keepdim=True)

        return pot


if __name__ == "__main__":
    mos = LJ()

    # 0.31,  # H
    # 0.66,  # O

    r1 = torch.arange(0.5, 2.5, step=0.02, requires_grad=True)

    nei = torch.full_like(r1, 1, dtype=torch.long)
    cen = torch.full_like(r1, 8, dtype=torch.long)

    r = r1
    lj = LJ()
    res = lj(cen, nei, r)
    re = res.mean()
    re.backward()

    nps = res.detach().numpy()
    r = r.detach().numpy()
    plt.plot(r, nps)
    plt.show()

    r1 = torch.arange(0.4, 1, step=0.02, requires_grad=True)
    r2 = torch.arange(0.4, 1, step=0.02, requires_grad=True)
    r3 = torch.arange(0.4, 1, step=0.02, requires_grad=True)

    r = torch.vstack((r1, r2, r3)).T
    nei = torch.full_like(r1, 1, dtype=torch.long)
    cen = torch.full_like(r1, 8, dtype=torch.long)

    r = r
    lj = LJ()
    res = lj(cen, nei, r)
    re = res.mean()
    re.backward()

    nps = res.detach().numpy()
    r = r.detach().numpy()
    plt.plot(r, nps[:, 0])
    plt.show()
