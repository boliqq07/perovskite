"""
copy from zhengl0217:Perovskite-Electronic-Structure-Feature-Python
https://github.com/zhengl0217/Perovskite-Electronic-Structure-Feature-Python/blob/master/compositional_descriptor.py
"""
import warnings

import numpy as np

from pymatgen.core import Element

try:
    import featurebox
    from featurebox.featurizers.atom.mapper import AtomPymatgenPropMap
    from featurebox.featurizers.state.statistics import WeightedAverage

except ImportError as e:
    warnings.warn("please `pip install featurebox` for more tools", ImportWarning)
    raise e


def T_f(rA, rB, rX):
    """
    公差系数。

    Parameters
    ----------
    rA:
        A位离子半径
    rB:
        B位离子半径
    rX:
        X位原子的离子半径
    """
    t_f = (rA + rX) / (np.sqrt(2) * (rB + rX))
    return t_f


def T_f_C_organic(rA, rB, rX, rXh):
    """
    公差系数,C-site 有机。

    Parameters
    ----------
    rA:
        A位离子半径
    rB:
        B位离子半径
    rX:
        X位原子的离子半径
    """
    t_f = (rA + rX) / (np.sqrt(2) * (rB + 0.5*rXh))
    return t_f

def O_f(rB, rX):
    """
    八面体因子

    Parameters
    ----------
    rB:
        B位离子半径
    rX:
        X位原子的离子半径

    Returns
    -------

    """
    # rB,rX分别表示B位和X位原子的离子半径
    # Of 八面体因子
    o_f = rB / rX

    return o_f


def r_sp(s, p):
    """
    s和p轨道半径之和(B, X)

    Parameters
    ----------
    s:
        s 轨道半径
    p:
        轨道半径

    """
    r_sp2 = s + p
    return r_sp2


def elemental_feature(A, B, C):
    """Not for molecule"""
    data = [A, B, C]
    for i in range(len(data)):
        if isinstance(data[i], dict):
            data[i] = [data[i], ]
    A, B, C = data

    func_map = [
        "common_oxidation_states"
        "average_ionic_radius",
        "average_cationic_radius",
    ]
    appma = AtomPymatgenPropMap(prop_name=func_map, search_tp="name")
    wa = WeightedAverage(appma, n_jobs=1, return_type="df")
    wa.set_feature_labels(["ave_A_" + i for i in appma.feature_labels])
    dA2 = wa.fit_transform(A)
    wa.set_feature_labels(["ave_B_" + i for i in appma.feature_labels])
    dB2 = wa.fit_transform(B)
    wa.set_feature_labels(["ave_C_" + i for i in appma.feature_labels])
    dC2 = wa.fit_transform(C)

    fff17 = dA2["ave_A_average_ionic_radius"]
    fff18 = dB2["ave_B_average_ionic_radius"]
    fff19 = dC2["ave_C_average_ionic_radius"]
    # Octahedral factor
    # Tolerance factor
    fff_22 = O_f(fff17, fff18)
    fff_23 = T_f(fff17, fff18, fff19)

    # A/B ion oxidation state
    ff_24 = (dA2["ave_A_common_oxidation_states"][0]) / (dB2["ave_B_common_oxidation_states"][0])
    # ionic_radius ratios
    ionic_ration_AO = fff17 / fff19
    ionic_ration_BO = fff18 / fff19

    return fff_22, fff_23, ff_24, ionic_ration_AO, ionic_ration_BO


def generalized_mean(x, p, N):
    """Generalized mean function capture the mean value of atomic properties by considering the ratio of each element in the structure.
    Args:
       x (np.ndarray): array of atomic properties for each atom in the structure.
       p (int): power parameter, e.g., harmonic mean (-1), geometric mean(0), arithmetic mean(1), quadratic mean(2).
       N (int): total number of atoms in the structure.
    Returns:
       float: generalized mean value.
    """
    if p != 0:
        D = 1 / (N)
        out = (D * sum(x ** p)) ** (1 / p)
    else:
        D = 1 / (N)
        out = np.exp(D * sum(np.log(x)))
    return out


def geometric_descriptor(element_dict):
    """Extract geometric mean of the atomic properties in the perovskite structure.
    Args:
       element_dict (dict): element frequency library in a perovskite structure, e.g., {'La': 4, 'Ba': 4, 'Co': 8, 'O': 24}
    Returns:
       np.ndarray: geometric based descriptor, including atomic_radius,  mendeleev number', common_oxidation_states, Pauling electronegativity, thermal_conductivity, average_ionic_radius, atomic_orbitals.
    """
    # encode the orbital types
    category = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
    # total number of atoms in a perovskite structure
    N = sum(element_dict.values())
    # obtain array of atomic properties for each element type
    atomic_number_list = []
    atomic_mass_list = []
    atomic_radius_list = []
    mendeleev_no_list = []
    common_oxidation_states_list = []
    Pauling_electronegativity_list = []
    row_list = []
    group_list = []
    block_list = []
    thermal_conductivity_list = []
    boiling_point_list = []
    melting_point_list = []
    average_ionic_radius_list = []
    molar_volume_list = []
    atomic_orbitals_list = []
    for item in element_dict:
        # extract atomic property from pymatgen
        ele = Element(item)
        atomic_number = ele.Z
        atomic_mass = float(str(ele.atomic_mass)[:-4])
        atomic_radius = float(str(ele.atomic_radius)[:-4])
        mendeleev_no = ele.mendeleev_no
        common_oxidation_states = ele.common_oxidation_states[0]
        Pauling_electronegativity = ele.X
        row = ele.row
        group = ele.group
        block = ele.block
        thermal_conductivity = float(str(ele.thermal_conductivity)[:-12])
        boiling_point = float(str(ele.boiling_point)[: -2])
        melting_point = float(str(ele.melting_point)[: -2])
        average_ionic_radius = float(str(ele.average_ionic_radius)[:-4])
        molar_volume = float(str(ele.molar_volume)[: -5])
        if '6s' in ele.atomic_orbitals.keys():
            atomic_orbitals = ele.atomic_orbitals['6s']
        elif '4s' in ele.atomic_orbitals.keys():
            atomic_orbitals = ele.atomic_orbitals['4s']
        else:
            atomic_orbitals = ele.atomic_orbitals['2s']
        # calculate the array of atomic properties for all atoms
        atomic_number_list += [atomic_number] * element_dict[item]
        atomic_mass_list += [atomic_mass] * element_dict[item]
        atomic_radius_list += [atomic_radius] * element_dict[item]
        mendeleev_no_list += [mendeleev_no] * element_dict[item]
        common_oxidation_states_list += [common_oxidation_states] * element_dict[item]
        Pauling_electronegativity_list += [Pauling_electronegativity] * element_dict[item]
        row_list += [row] * element_dict[item]
        group_list += [group] * element_dict[item]
        block_list += [category[block]] * element_dict[item]
        thermal_conductivity_list += [thermal_conductivity] * element_dict[item]
        boiling_point_list += [boiling_point] * element_dict[item]
        melting_point_list += [melting_point] * element_dict[item]
        average_ionic_radius_list += [average_ionic_radius] * element_dict[item]
        molar_volume_list += [molar_volume] * element_dict[item]
        atomic_orbitals_list += [atomic_orbitals] * element_dict[item]
    return [generalized_mean(np.array(atomic_number_list), 1, N)] + [
        generalized_mean(np.array(atomic_radius_list), 1, N)] + [
               generalized_mean(np.array(mendeleev_no_list), 1, N)] + [
               generalized_mean(np.array(common_oxidation_states_list), 1, N)] + [
               generalized_mean(np.array(Pauling_electronegativity_list), 1, N)] + [
               generalized_mean(np.array(thermal_conductivity_list), 1, N)] + [
               generalized_mean(np.array(average_ionic_radius_list), 1, N)] + [
               generalized_mean(np.array(atomic_orbitals_list), 1, N)]


if __name__ == "__main__":
    print('LaBaCo2O6',
          geometric_descriptor({'La': 4, 'Ba': 4, 'Co': 8, 'O': 24}))
