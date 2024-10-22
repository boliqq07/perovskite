"""For quick use.

Examples:
    >>> atoms = aaa.get_atoms(structure=structure)
    >>> structure = aaa.get_structure(atoms=atoms)
"""

from pymatgen.io.ase import AseAtomsAdaptor

aaa = AseAtomsAdaptor()
