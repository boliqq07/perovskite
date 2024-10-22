"""Pure torch functions."""

import math

import torch
from torch_geometric.nn import radius_graph
# from torch_cluster import radius_graph
from torch_geometric.utils import remove_self_loops
from torch_scatter import segment_coo, segment_csr


def add_edge_total(data, use_pbc=True, use_exist_index=True, **kwargs):
    """Total function to add edge_weight,edge_attr,data.edge_index."""
    if not use_pbc:
        if hasattr(data, "edge_index") and use_exist_index:
            data = add_edge_no_pbc_from_index(data, **kwargs)
        else:
            data = add_edge_no_pbc(data, **kwargs)
    else:
        if hasattr(data, "edge_index") and use_exist_index:
            data = add_edge_pbc_from_index(data, **kwargs)
        else:
            data = add_edge_pbc(data, **kwargs)
    return data


def add_edge_no_pbc(data, cutoff=5.0):
    """Add edge_weight,edge_attr,data.edge_index ignore pbc."""
    edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, flow='source_to_target')
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index[0], edge_index[1]

    distance_vectors = data.pos[row] - data.pos[col]

    edge_weight = distance_vectors.norm(dim=-1)

    data.edge_attr = distance_vectors
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    return data


def add_edge_no_pbc_from_index(data, dmin=0.0, dmax=6.0):
    """Add edge_weight,edge_attr ignore pbc from the old edge_index."""

    edge_index = data.edge_index

    row, col = edge_index[0], edge_index[1]

    distance_vectors = data.pos[row] - data.pos[col]

    edge_weight = distance_vectors.norm(dim=-1)

    data.edge_index = edge_index
    data.edge_weight = edge_weight
    data.edge_attr = distance_vectors

    return data


def compute_neighbors(data):
    """Add neighbors for each atom"""
    # Get number of neighbors
    # segment_coo assumes sorted index
    edge_index = data.edge_index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(
        ones, edge_index[1], dim_size=data.natoms.sum()
    )

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    data.neighbors = neighbors
    return data


def add_edge_pbc_from_index(data, **kwargs):
    """Add edge_weight,edge_attr with pbc from the old edge_index."""

    if not hasattr(data, "neighbors"):
        data = compute_neighbors(data)
    cell = data.cell
    edge_index = data.edge_index
    neighbors = data.neighbors
    pos = data.pos
    cell_offsets = data.cell_offsets

    edge_index, edge_weight, cell_offsets, distance_vectors = _add_edge_pbc_from_index(pos,
                                                                                       edge_index,
                                                                                       cell,
                                                                                       cell_offsets,
                                                                                       neighbors,
                                                                                       )

    data.cell_offsets = cell_offsets
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    data.edge_attr = distance_vectors

    return data


def add_edge_pbc(data, cutoff=5.0, max_neighbors=16):
    """Add edge_weight,edge_attr with pbc."""
    edge_index, cell_offsets, neighbors = _radius_graph_pbc(
        data, cutoff, max_neighbors
    )

    edge_index, edge_weight, cell_offsets, distance_vectors = _add_edge_pbc_from_index(
        data.pos,
        edge_index,
        data.cell,
        cell_offsets,
        neighbors,
    )

    data.edge_attr = distance_vectors
    data.cell_offsets = cell_offsets
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    return data


def _add_edge_pbc_from_index(pos,
                             edge_index,
                             cell,
                             cell_offsets,
                             neighbors,
                             ):
    device = cell.device
    row, col = edge_index
    distance_vectors = pos[row] - pos[col]
    neighbors = neighbors.to(device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=device)[distances != 0]

    edge_index = edge_index[:, nonzero_idx]
    edge_weight = distances[nonzero_idx]
    cell_offsets = cell_offsets[nonzero_idx]

    return edge_index, edge_weight, cell_offsets, distance_vectors


def _get_max_neighbors_mask(
        natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
            max_num_neighbors <= max_num_neighbors_threshold
            or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], torch.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
            index * max_num_neighbors
            + torch.arange(len(index), device=device)
            - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def _radius_graph_pbc(data, radius, max_num_neighbors_threshold, pbc=None):
    if pbc is None:
        pbc = [True, True, True]
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
            torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
                 torch.div(
                     atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
                 )
             ) + index_offset_expand
    index2 = (
                     atom_count_sqr % num_atoms_per_image_expand
             ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = _get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def distribute_edge(data, r_cs, r_c):
    """For data.edge_attr is the x,y,z site of atom."""
    assert hasattr(data, "edge_attr") and data.edge_attr is not None
    assert data.edge_attr.shape[1] == 3
    if not hasattr(data, "edge_weight") or data.edge_weight is None:
        data.edge_weight = torch.linalg.norm(data.edge_attr, dim=1)

    wei = data.edge_weight

    attr = torch.cat((wei.reshape(-1, 1), data.edge_attr), dim=1)

    sr = torch.clone(attr[:, 0])
    r_m1 = sr <= r_cs
    r_m3 = sr >= r_c
    r_m2 = ~(r_m3 | r_m1)

    sr[r_m1] = 1 / sr[r_m1]

    u = (sr[r_m2] - r_cs) / (r_c - r_cs)
    sr[r_m2] = 1 / sr[r_m2] * (0.5 * torch.cos(math.pi * u) + 0.5)  # from deep Potential
    sr[r_m3] = 0

    attr = attr / attr[:, 0].reshape(-1, 1) * sr.reshape(-1, 1)

    data.edge_attr = attr
    data.edge_weight = sr
    return data
