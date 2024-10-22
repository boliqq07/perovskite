# -*- coding: utf-8 -*-
# @Time  : 2022/9/27 15:12
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import ReLU


class Morse(torch.nn.Module):
    """Morse potential"""

    def __init__(self):
        super().__init__()
        from ase.data import covalent_radii
        radii = torch.from_numpy(covalent_radii).float()
        self.lin1 = nn.Sequential(
            nn.Linear(2, 8, bias=True), ReLU(),
            nn.Linear(8, 8, bias=True), ReLU(),
            nn.Linear(8, 8, bias=True)
        )
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

    def _forward_r(self, cen, nei, r):

        r0 = (self.radii[nei] + self.radii[cen]).view(-1, 1)
        param_in = torch.vstack((nei + cen, nei * cen)).T.float()
        param_out = self.lin1(param_in)
        # param_out = torch.abs(param_out) # not use

        D, k, c, alpha, n_3, n1, n2, _ = torch.chunk(param_out, chunks=8, dim=1)

        D = torch.abs(D)
        k = torch.abs(k)
        n1 = torch.abs(n1)

        n1 = 0.001 * n1
        n2 = 0.001 * n2
        n_3 = 0.001 * n_3

        alpha = torch.minimum(alpha, torch.full_like(alpha, 20))
        alpha = torch.maximum(alpha, torch.full_like(alpha, -20))
        k = torch.minimum(k, torch.full_like(alpha, 20))

        MV_COEF = 1.0
        LINEBASE_COEF = 2.0

        alpha = (1 + 0.01 * alpha) * MV_COEF
        k = (0.8 + 0.01 * k) * LINEBASE_COEF

        core = -k * (r - alpha * r0)

        pot = D * torch.pow(1 - torch.exp(core), 2) + n_3 * r ** (1 / 3) + n2 * r ** 2

        pot = torch.minimum(pot, torch.full_like(pot, 10))

        pot = pot - n1 * r + c

        return pot

    def _forward_xyz(self, cen, nei, r):

        r1 = torch.sum(r ** 2, dim=1, keepdim=True)

        pot = self._forward_r(cen, nei, r1)

        r2 = r * torch.abs(r)

        beta = r2 / r1

        pot = beta * pot

        return pot


if __name__ == "__main__":
    mos = Morse()

    # 0.31,  # H
    # 0.66,  # O

    r1 = torch.arange(0.2, 5, step=0.02, requires_grad=True)

    a = torch.full_like(r1, 1, dtype=torch.long)
    b = torch.full_like(r1, 8, dtype=torch.long)

    r = r1
    lj = Morse()
    res = lj(a, b, r)
    re = res.mean()
    re.backward()

    nps = res.detach().numpy()
    r = r.detach().numpy()
    plt.plot(r, nps)
    plt.show()

    r1 = torch.arange(0.10, 1.7, step=0.02, requires_grad=True)
    r2 = torch.arange(0.10, 1.7, step=0.02, requires_grad=True)
    r3 = torch.arange(0.10, 1.7, step=0.02, requires_grad=True)

    r = torch.vstack((r1, r2, r3)).T
    a = torch.full_like(r1, 1, dtype=torch.long)
    b = torch.full_like(r1, 8, dtype=torch.long)

    r = r
    lj = Morse()
    res = lj(a, b, r)
    re = res.mean()
    re.backward()

    nps = res.detach().numpy()

    r = r.detach().numpy()
    plt.plot(r, nps[:, 0])
    plt.show()

    plt.plot(r, nps[:, 1])
    plt.show()

    plt.plot(r, nps[:, 2])
    plt.show()
