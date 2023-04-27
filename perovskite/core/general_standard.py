
# Use Asap for a huge performance increase if it is installed
from ase.spacegroup import crystal
from ase.build import cut

# #######221##################0

def perovskite_conv_221(a_atom="Ba", b_atom="Ti", c_atom="O", a=7, size=(1, 1, 1)):
    a_atom = [a_atom] * 2 if not isinstance(a_atom, list) else a_atom
    b_atom = [b_atom] * 2 if not isinstance(b_atom, list) else b_atom
    c_atom = [c_atom] * 6 if not isinstance(c_atom, list) else c_atom

    place = [
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],

        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.0],
        [0.25, 0.75, 0.0],
        [0.75, 0.25, 0.0],
        [0.75, 0.75, 0.0],

    ]
    ats = a_atom + b_atom + c_atom

    atoms = crystal(ats, place, size=size,
                    cellpar=[a, a, a / 1.4142, 90, 90, 90])
    return atoms


def perovskite_prim_221(a_atom="Ba", b_atom="Ti", c_atom="O", a=7, size=(1, 1, 1)):
    a_atom = [a_atom] * 1 if not isinstance(a_atom, list) else a_atom
    b_atom = [b_atom] * 1 if not isinstance(b_atom, list) else b_atom
    c_atom = [c_atom] * 1 if not isinstance(c_atom, list) else c_atom

    place = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
    ]
    ats = a_atom + b_atom + c_atom

    atoms = crystal(ats, place, size=size, spacegroup=221,
                    cellpar=[a, a, a, 90, 90, 90])
    return atoms


# # 轴拉长
# ########127##################
def perovskite_conv_127(a_atom="Ba", b_atom="Ti", c_atom="O", a=7, alpha=1.1, cycle=0.05, size=(1, 1, 1)):
    a_atom = [a_atom] * 2 if not isinstance(a_atom, list) else a_atom
    b_atom = [b_atom] * 2 if not isinstance(b_atom, list) else b_atom
    c_atom = [c_atom] * 6 if not isinstance(c_atom, list) else c_atom

    place = [
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],

        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],

        [0.25 + cycle, 0.25 - cycle, 0.0],
        [0.25 - cycle, 0.75 - cycle, 0.0],
        [0.75 + cycle, 0.25 + cycle, 0.0],
        [0.75 - cycle, 0.75 + cycle, 0.0],
    ]
    ats = a_atom + b_atom + c_atom

    atoms = crystal(ats, place, size=size,
                    cellpar=[a, a, a / 1.4142 * alpha, 90, 90, 90])
    return atoms


def perovskite_prim_127(a_atom="Ba", b_atom="Ti", c_atom="O", a=7, alpha=1.0, cycle=0.04, size=(1, 1, 1)):
    a_atom = [a_atom] * 1 if not isinstance(a_atom, list) else a_atom
    b_atom = [b_atom] * 1 if not isinstance(b_atom, list) else b_atom
    c_atom = [c_atom] * 2 if not isinstance(c_atom, list) else c_atom

    place = [
        [0.0, 0.5, 0],
        [0.0, 0.0, 0.5],
        [0.25 - cycle, 0.25 + cycle, 0.50, ],  # cycle
        [0.0, 0.0, 0.0],  # bond
    ]
    ats = a_atom + b_atom + c_atom

    atoms = crystal(ats, place, size=size, spacegroup=127,
                    cellpar=[a, a, a / 1.4142 * alpha, 90, 90, 90])
    return atoms


# #####62,################################

def perovskite_conv_62(a_atom="Ca", b_atom="Ti", c_atom="O", a=7, alpha=1.0, move=0.04, cycle=0.04, size=(1, 1, 1)):
    a_atom = [a_atom] * 1 if not isinstance(a_atom, list) else a_atom
    b_atom = [b_atom] * 1 if not isinstance(b_atom, list) else b_atom
    c_atom = [c_atom] * 2 if not isinstance(c_atom, list) else c_atom

    ats = a_atom + b_atom + c_atom

    place = [
        [0.5 - move, 0.25, 0],
        [0.0, 0.0, 0],
        [0.25 - cycle, 0.0, 0.25 + cycle],  # cycle
        [0.5 + move / 2, 0.25, 0.57 + move],  # bond
    ]

    # ["Ca", "Ti", "O", "O", ],
    atoms = crystal(ats, place, size=(2, 1, 2), spacegroup=62,
                    cellpar=[a, a * 1.4142 * alpha, a, 90, 90, 90], primitive_cell=True)

    atoms2 = cut(atoms, a=(0.5, 0, 0), b=(0, 1, 0), c=(0, 0, 0.5), origo=(0.25, 0, 0.25))
    # change dimension
    position = atoms2.get_positions()
    atoms2.set_positions(position[:, (0, 2, 1)])
    atoms2.set_cell(atoms2.get_cell()[:, (0, 2, 1)])
    atoms2.set_cell(atoms2.get_cell()[(0, 2, 1), :])
    atoms2 *= size

    return atoms2


def perovskite_prim_62(a_atom="Ca", b_atom="Ti", c_atom="O", a=7, alpha=1.0, move=0.04, cycle=0.04, size=(1, 1, 1)):
    a_atom = [a_atom] * 1 if not isinstance(a_atom, list) else a_atom
    b_atom = [b_atom] * 1 if not isinstance(b_atom, list) else b_atom
    c_atom = [c_atom] * 2 if not isinstance(c_atom, list) else c_atom

    place = [
        [0.5 - move, 0.25, 0],
        [0.0, 0.0, 0],
        [0.25 - cycle, 0.0, 0.25 + cycle],  # cycle
        [0.5 + move / 2, 0.25, 0.57 + move],  # bond
    ]
    ats = a_atom + b_atom + c_atom
    # ["Ca", "Ti", "O", "O", ],
    atoms = crystal(ats, place, size=size, spacegroup=62,
                    cellpar=[a, a * 1.4142 * alpha, a, 90, 90, 90], primitive_cell=True)
    return atoms


if __name__ == "__main__":
    at = perovskite_prim_221()
    at.rotate(90, (1, 1, 0), rotate_cell=True)
    sg = at.info["spacegroup"]

    prim_cell = sg.scaled_primitive_cell

    # Preserve calculator if present:

    atoms = cut(at, a=prim_cell[0], b=prim_cell[1], c=prim_cell[2])

    from ase.visualize import view

    view(atoms)