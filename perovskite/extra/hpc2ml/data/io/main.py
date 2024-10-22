"""
Examples for ase db:
    >>> from hpc2ml.data.structuretodata import StructureToData
    >>> addsap = StructureToData()
    >>> source_path = "."
    >>> res = addsap.sparse_source_data(source_file="data.db", source_path=source_path, fmt="ase",)

Examples for Vasp:
    >>> from hpc2ml.data.structuretodata import StructureToData
    >>> addsap = StructureToData()
    >>> source_path = find_leaf_path("./data") # get path list
    >>> res = addsap.sparse_source_data(source_file="vasprun.xml",
    ... source_path=source_path, fmt="vasprun",)
    >>> res2 = addsap.sparse_source_data(source_file="vasprun.xml",
    ... source_path=source_path, fmt="vasprun_traj",space=5)

Examples for csv:
    >>> from hpc2ml.data.structuretodata import StructureToData
    >>> addsap = StructureToData()
    >>> source_path = "."
    >>> res = addsap.sparse_source_data(source_file="tb.csv", source_path=source_path, fmt="csv",)

"""

import os
from typing import Union, Sequence

from mgetool.tool import parallelize


def sparse_source_data(source_file: str = "vasprun.xml", source_path: Union[Sequence, str] = ".",
                       fmt: str = "vasprun",
                       n_jobs=4, **kwargs):
    """
    Sparse data by different function.

    Args:
        source_file: file name to sparse.
        source_path: (str,list), path or paths.
        fmt: str, load function named  "sparse_{fmt}" in from ``hpc2ml.data.io`` .
        n_jobs: int, the parallel number to load,
        **kwargs: dict, the parameter in "sparse_{fmt}".

    Returns:
        data_dict:dict, data
    """
    if fmt == "auto":
        suffix = source_file.split(".")[-1]
        fmt = {"xml": "vasprun", "db": "ase", "csv": "csv"}[suffix]

    if isinstance(source_path, str):
        source_path = [source_path, ]

    source_file = [os.path.join(pathi, source_file) for pathi in source_path]
    from hpc2ml.data import io
    func = getattr(io, f"sparse_{fmt}")

    def func2(i):
        try:
            dicti = func(i, **kwargs)
        except:
            dicti = {}
            print(f"Error for : {i}")
        return dicti

    dct = {}
    res = parallelize(n_jobs=n_jobs, func=func2, iterable=source_file, tq=True, desc="sparse the source data")
    [dct.update(i) for i in res]

    # for i in source_file:
    #     try:
    #         dcti = func(i)
    #         dct.update(dcti)
    #     except:
    #         print(f"Error: {i}.")

    return dct
