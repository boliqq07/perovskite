import numpy as np
from pymatgen.io.vasp import Vasprun


def sparse_vasprun(path: str, forces=True):
    """Sparse vasprun.xml and load finall data."""
    vr = Vasprun(path)
    steps = vr.ionic_steps
    sym = [i.split(" ")[1].split("_")[0] for i in vr.potcar_symbols]
    potcar_label = tuple([vr.potcar_symbols[sym.index(i)] for i in vr.atomic_symbols])

    if forces:
        stepi = steps[-1]
        dct = {f"{path}": {"structure": stepi["structure"], "forces": np.array(stepi["forces"]),
                           "stress": np.array(stepi["stress"]),
                           "energy": stepi["e_wo_entrp"], "potcar_symbol": potcar_label, "incar": vr.incar,
                           "kpoints_kpts": vr.kpoints.kpts, "kpoints_style": vr.kpoints.style.name,
                           "efermi": vr.efermi}}
    else:
        dct = {f"{path}": {"structure": vr.final_structure, "energy": vr.final_energy, "potcar_symbol": potcar_label,
                           "incar": vr.incar, "kpoints_kpts": vr.kpoints.kpts, "kpoints_style": vr.kpoints.style.name,
                           "efermi": vr.efermi}}

    return dct


def sparse_vasprun_traj(path: str, space=20):
    """Sparse vasprun.xml and load all ion step, where space is the sample interval."""
    vr = Vasprun(path)
    steps = vr.ionic_steps
    sym = [i.split(" ")[1].split("_")[0] for i in vr.potcar_symbols]
    potcar_label = tuple([vr.potcar_symbols[sym.index(i)] for i in vr.atomic_symbols])

    dct = {}
    for i, stepi in enumerate(steps):
        if i % space == 0 and i != len(steps) - 1:
            dcti = {"structure": stepi["structure"], "forces": np.array(stepi["forces"]),
                    "stress": np.array(stepi["stress"]),
                    "energy": stepi["e_wo_entrp"], "potcar_symbol": potcar_label, "incar": vr.incar,
                    "kpoints_kpts": vr.kpoints.kpts, "kpoints_style": vr.kpoints.style.name, "efermi": vr.efermi}
            dct.update({f"{path}_step{i}": dcti})

    stepi = steps[-1]
    dctn1 = {f"{path}": {"structure": stepi["structure"], "forces": np.array(stepi["forces"]),
                         "stress": np.array(stepi["stress"]),
                         "energy": stepi["e_wo_entrp"], "potcar_symbol": potcar_label, "incar": vr.incar,
                         "kpoints_kpts": vr.kpoints.kpts, "kpoints_style": vr.kpoints.style.name, "efermi": vr.efermi}}
    dct.update(dctn1)
    return dct


if __name__ == "__main__":
    # path = r"C:\Users\98679\PycharmProjects\hpc2ml\test\test_data\pure_opt\vasprun.xml"
    path = r"../../../../hpc2ml/test/test_data/pure_opt/vasprun.xml"
    path = r"../../../../hpc2ml/test/test_data/OH_add_static/vasprun.xml"
    res = sparse_vasprun_traj(path)
    res2 = sparse_vasprun(path, forces=False)
