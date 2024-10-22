"""
Examples:
    >>> data = pd.read_pickle("st_and_energy.pkl_pd")
    >>> ks, (structure, energy), k2s = unpack(data)
    >>> data1 = pack(ks, (structure, energy), k2s)
"""

from typing import List, Sequence, Union

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm


def pack(ks: Union[Sequence, np.ndarray], vs: Union[Sequence, np.ndarray],
         k2s: Union[Sequence, np.ndarray] = None, trans: bool = True):
    """Pack the array to dict."""

    res4 = pd.DataFrame.from_dict({i: j for i, j in zip(ks, vs)})

    if k2s is None:
        pass
    else:
        res4.index = k2s

    if trans:
        res4 = res4.T

    data = res4.to_dict()
    return data


def unpack(data: dict, trans=True):
    """Unpack the dict to names and value."""
    res = pd.DataFrame.from_dict(data)
    if trans:
        res = res.T
    ks = res.columns.values
    vs = res.values
    vs = [vs[:, i] for i in range(vs.shape[1])]
    k2s = res.index.values
    return ks, vs, k2s


def unpack2dict(data: dict, trans=True):
    """Un-squeeze the 2 layer dict to simple data."""
    ks, vs, k2s = unpack(data, trans=trans)
    data_new = {ksi: vsi for ksi, vsi in zip(ks, vs)}
    return data_new, k2s


def couple_org(structures: List[Structure], energies: Sequence = None, **kwargs):
    """Group (*,*H) structure by formula."""
    if energies is None:
        energies = [np.nan] * len(structures)
    kwargs2 = {}
    kwargs2.update({"energies": energies})
    kwargs2.update(kwargs)

    # name = list(kwargs.keys())
    values = list(kwargs2.values())

    ans = [set(si.composition.formula.split(" ")) for si in structures]
    l_ans = len(ans)
    st_dict = [(si, ni) for si, ni in zip(structures, range(l_ans))]

    couple = [[] for _ in range(1 + len(values))]

    for i in tqdm(range(l_ans)):
        for j in range(i + 1, l_ans):
            if ans[i].issubset(ans[j]) and len(ans[j] - ans[i]) == 1:
                st1 = st_dict[i][0]
                st2 = st_dict[j][0]
                c2 = st2.cart_coords[:-1]
                c1 = st1.cart_coords
                if len(st2.atomic_numbers) - len(st1.atomic_numbers) == 1:
                    if np.sum(np.abs(c2 - c1)) / len(c1) < 100 and np.sum(np.abs(c2 - c1)) / len(c1) > 1:  # ????
                        # diff = np.sum(np.abs(c2 - c1)) / len(c1) # structure difference
                        couple[0].extend((st_dict[i][0], st_dict[j][0]))
                        for ni, vi in enumerate(values):
                            couple[ni + 1].extend((vi[i], vi[j]))

    return couple


if __name__ == "__main__":
    data = pd.read_pickle("st_and_energy.pkl_pd")
    ks, (structure, energy), k2s = unpack(data)
    data1 = pack(ks, (structure, energy), k2s)
