import itertools
from typing import Literal, Sequence

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator, AbstractComparator
from pymatgen.core.composition import SpeciesLike
from tqdm import tqdm


class StructureMatcherExtend(StructureMatcher):
    """
    Class to match structures by similarity.

    Algorithm:

    1. Given two structures: s1 and s2
    2. Optional: Reduce to primitive cells.
    3. If the number of sites do not match, return False
    4. Reduce to s1 and s2 to Niggli Cells
    5. Optional: Scale s1 and s2 to same volume.
    6. Optional: Remove oxidation states associated with sites
    7. Find all possible lattice vectors for s2 within shell of ltol.
    8. For s1, translate an atom in the smallest set to the origin
    9. For s2: find all valid lattices from permutations of the list
       of lattice vectors (invalid if: det(Lattice Matrix) < half
       volume of original s2 lattice)
    10. For each valid lattice:

        a. If the lattice angles of are within tolerance of s1,
           basis change s2 into new lattice.
        b. For each atom in the smallest set of s2:

            i. Translate to origin and compare fractional sites in
            structure within a fractional tolerance.
            ii. If true:

                ia. Convert both lattices to Cartesian and place
                both structures on an average lattice
                ib. Compute and return the average and max rms
                displacement between the two structures normalized
                by the average free length per atom

                if fit function called:
                    if normalized max rms displacement is less than
                    stol. Return True

                if get_rms_dist function called:
                    if normalized average rms displacement is less
                    than the stored rms displacement, store and
                    continue. (This function will search all possible
                    lattices for the smallest average rms displacement
                    between the two structures)
    """

    def __init__(
            self,
            ltol: float = 0.2,
            stol: float = 0.3,
            angle_tol: float = 5,
            primitive_cell: bool = True,
            scale: bool = True,
            attempt_supercell: bool = False,
            allow_subset: bool = False,
            comparator: AbstractComparator = None,
            supercell_size: Literal["num_sites", "num_atoms", "volume"] = "num_sites",
            ignored_species: Sequence[SpeciesLike] = (),
    ):
        """
        Args:
            ltol (float): Fractional length tolerance. Default is 0.2.
            stol (float): Site tolerance. Defined as the fraction of the
                average free length per atom := ( V / Nsites ) ** (1/3)
                Default is 0.3.
            angle_tol (float): Angle tolerance in degrees. Default is 5 degrees.
            primitive_cell (bool): If true: input structures will be reduced to
                primitive cells prior to matching. Default to True.
            scale (bool): Input structures are scaled to equivalent volume if
               true; For exact matching, set to False.
            attempt_supercell (bool): If set to True and number of sites in
                cells differ after a primitive cell reduction (divisible by an
                integer) attempts to generate a supercell transformation of the
                smaller cell which is equivalent to the larger structure.
            allow_subset (bool): Allow one structure to match to the subset of
                another structure. Eg. Matching of an ordered structure onto a
                disordered one, or matching a delithiated to a lithiated
                structure. This option cannot be combined with
                attempt_supercell, or with structure grouping.
            comparator (Comparator): A comparator object implementing an equals
                method that declares equivalency of sites. Default is
                SpeciesComparator, which implies rigid species
                mapping, i.e., Fe2+ only matches Fe2+ and not Fe3+.

                Other comparators are provided, e.g., ElementComparator which
                matches only the elements and not the species.

                The reason why a comparator object is used instead of
                supplying a comparison function is that it is not possible to
                pickle a function, which makes it otherwise difficult to use
                StructureMatcher with Python's multiprocessing.
            supercell_size (str or list): Method to use for determining the
                size of a supercell (if applicable). Possible values are
                'num_sites', 'num_atoms', 'volume', or an element or list of elements
                present in both structures.
            ignored_species (list): A list of ions to be ignored in matching.
                Useful for matching structures that have similar frameworks
                except for certain ions, e.g., Li-ion intercalation frameworks.
                This is more useful than allow_subset because it allows better
                control over what species are ignored in the matching.
        """
        if comparator is None:
            comparator = ElementComparator()
        super(StructureMatcherExtend, self).__init__(ltol=ltol,
                                                     stol=stol, scale=scale,
                                                     angle_tol=angle_tol,
                                                     primitive_cell=primitive_cell,
                                                     attempt_supercell=attempt_supercell,
                                                     allow_subset=allow_subset,
                                                     comparator=comparator,
                                                     supercell_size=supercell_size,
                                                     ignored_species=ignored_species)

    def group_structures_and_index(self, s_list, anonymous=False):

        """
        Given a list of structures, use fit to group
        them by structural equality.

        Args:
            s_list ([Structure]): List of structures to be grouped
            anonymous (bool): Whether to use anonymous mode.

        Returns:
            A list of lists of matched structures
            Assumption: if s1 == s2 but s1 != s3, than s2 and s3 will be put
            in different groups without comparison.
        """
        if self._subset:
            raise ValueError("allow_subset cannot be used with group_structures")

        original_s_list = list(s_list)
        s_list = self._process_species(s_list)
        # Prepare reduced structures beforehand
        s_list = [self._get_reduced_structure(s, self._primitive_cell, niggli=True) for s in s_list]

        # Use structure hash to pre-group structures
        if anonymous:

            def c_hash(c):
                return c.anonymized_formula

        else:
            c_hash = self._comparator.get_hash

        def s_hash(s):
            return c_hash(s[1].composition)

        sorted_s_list = sorted(enumerate(s_list), key=s_hash)
        all_groups = []
        index_groups = []

        # For each pre-grouped list of structures, perform actual matching.
        for _, g in tqdm(itertools.groupby(sorted_s_list, key=s_hash)):
            unmatched = list(g)
            while len(unmatched) > 0:
                i, refs = unmatched.pop(0)
                matches = [i]
                if anonymous:
                    inds = filter(
                        lambda i: self.fit_anonymous(refs, unmatched[i][1], skip_structure_reduction=True),
                        list(range(len(unmatched))),
                    )
                else:
                    inds = filter(
                        lambda i: self.fit(refs, unmatched[i][1], skip_structure_reduction=True),
                        list(range(len(unmatched))),
                    )
                inds = list(inds)
                matches.extend([unmatched[i][0] for i in inds])
                unmatched = [unmatched[i] for i in range(len(unmatched)) if i not in inds]
                all_groups.append([original_s_list[i] for i in matches])
                index_groups.append(matches)

        return all_groups, index_groups

    def select_min_in_group(self, structure, array):
        """group the structures, and select the min index in each group according the values in array."""
        gp, index = self.group_structures_and_index(structure)
        index = [np.array(i) for i in index]
        select_index = []
        # log = {}
        for indexi in index:
            if len(indexi) == 1:
                ini = indexi[0]
            else:
                gp_e = array[indexi]
                min_ind = np.argmin(gp_e)
                ini = indexi[min_ind]
                # sel_k = gp_k[min_ind]
                # log.update({sel_k: (tuple(gp_k), tuple(gp_e))})
            select_index.append(ini)
        return np.array(select_index)
