import pandas as pd
from pymatgen.core import Structure

from hpc2ml.data.io import aaa
from hpc2ml.db.ase_db_extension.ext_db_csv import _decode_dct


def sparse_csv(csv_file: str, fmt=lambda x: x):
    """convert table."""
    if isinstance(csv_file, str):
        file = pd.read_csv(csv_file, index_col=0)
    else:
        file = csv_file

    data = file.to_dict(orient="index")

    dct = {}
    for k in data.keys():
        value = _decode_dct(data[k])
        if "atoms" in value:
            st = aaa.get_structure(value["atoms"])
            value["structure"] = st
        elif "structure" in value:
            if isinstance(value["structure"], Structure):
                pass
            else:
                try:
                    st = Structure.from_str(value["structure"], fmt="json")
                except:
                    st = fmt(value["structure"])
                value["structure"] = st
        else:
            raise NotImplementedError("Can't defined structure by names : `atoms` or `structure` in csv")
        dct = {k: value}

    return dct
