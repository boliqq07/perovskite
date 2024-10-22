from typing import Dict, Any

_SPECIAL_IRREPS = [None]

from e3nn import o3


def _fix_irreps_dict(d: Dict[str, Any]):

    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


def _irreps_compatible(ir1: Dict[str, o3.Irreps], ir2: Dict[str, o3.Irreps]):

    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)
