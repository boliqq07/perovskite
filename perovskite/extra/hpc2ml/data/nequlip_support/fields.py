from typing import Tuple, Union, Sequence, Set

from hpc2ml.data.nequlip_support import AtomicDataDict

PBC = Union[bool, Tuple[bool, bool, bool]]

_DEFAULT_LONG_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.BATCH_KEY,
}
_DEFAULT_NODE_FIELDS: Set[str] = {
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.NODE_FEATURES_KEY,
    AtomicDataDict.NODE_ATTRS_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.BATCH_KEY,
}
_DEFAULT_EDGE_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
    AtomicDataDict.EDGE_VECTORS_KEY,
    AtomicDataDict.EDGE_LENGTH_KEY,
    AtomicDataDict.EDGE_ATTRS_KEY,
    AtomicDataDict.EDGE_EMBEDDING_KEY,
    AtomicDataDict.EDGE_FEATURES_KEY,
}
_DEFAULT_GRAPH_FIELDS: Set[str] = {
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.STRESS_KEY,
    AtomicDataDict.VIRIAL_KEY,
    AtomicDataDict.PBC_KEY,
    AtomicDataDict.CELL_KEY,
}
_NODE_FIELDS: Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS: Set[str] = set(_DEFAULT_EDGE_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_LONG_FIELDS: Set[str] = set(_DEFAULT_LONG_FIELDS)


def register_fields(
        node_fields: Sequence[str] = [],
        edge_fields: Sequence[str] = [],
        graph_fields: Sequence[str] = [],
        long_fields: Sequence[str] = [],
) -> None:
    r"""Register fields as being per-atom, per-edge, or per-frame.

    Args:
        node_permute_fields: fields that are equivariant to node permutations.
        edge_permute_fields: fields that are equivariant to edge permutations.
    """
    node_fields: set = set(node_fields)
    edge_fields: set = set(edge_fields)
    graph_fields: set = set(graph_fields)
    allfields = node_fields.union(edge_fields, graph_fields)
    assert len(allfields) == len(node_fields) + len(edge_fields) + len(graph_fields)
    _NODE_FIELDS.update(node_fields)
    _EDGE_FIELDS.update(edge_fields)
    _GRAPH_FIELDS.update(graph_fields)
    _LONG_FIELDS.update(long_fields)
    if len(set.union(_NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS)) < (
            len(_NODE_FIELDS) + len(_EDGE_FIELDS) + len(_GRAPH_FIELDS)
    ):
        raise ValueError(
            "At least one key was registered as more than one of node, edge, or graph!"
        )


def deregister_fields(*fields: Sequence[str]) -> None:
    r"""Deregister a field registered with ``register_fields``.

    Silently ignores fields that were never registered to begin with.

    Args:
        *fields: fields to deregister.
    """
    for f in fields:
        assert f not in _DEFAULT_NODE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_EDGE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_GRAPH_FIELDS, "Cannot deregister built-in field"
        _NODE_FIELDS.discard(f)
        _EDGE_FIELDS.discard(f)
        _GRAPH_FIELDS.discard(f)
