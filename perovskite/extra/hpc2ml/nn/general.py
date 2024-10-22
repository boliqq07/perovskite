"""
Operation utilities on lists and arrays, for torch,

Notes
    just for network.
"""

from collections import abc
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any
from typing import Union, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage
from torch_sparse import SparseTensor


def to_list(x: Union[abc.Iterable, np.ndarray]) -> List:
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, abc.Iterable):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()  # noqa
    return [x]


def fast_label_binarize(value: List, labels: List) -> List[int]:
    """Faster version of label binarize

    `label_binarize` from scikit-learn is slow when run 1 label at a time.
    `label_binarize` also is efficient for large numbers of classes, which is not
    common in `megnet`

    Args:
        value: Value to encode
        labels (list): Possible class values
    Returns:
        ([int]): List of integers
    """

    if len(labels) == 2:
        return [int(value == labels[0])]
    output = [0] * len(labels)
    if value in labels:
        output[labels.index(value)] = 1
    return output


def check_shape(array: Optional[np.ndarray], shape: Sequence) -> bool:
    """
    Check if array complies with shape. Shape is a sequence of
    integer that may end with None. If None is at the end of shape,
    then any shapes in array after that dimension will match with shape.

    Example: array with shape [10, 20, 30, 40] matches with [10, 20, None], but
        does not match with shape [10, 20, 30, 20]

    Args:
        array (np.ndarray or None): array to be checked
        shape (Sequence): integer array shape, it may ends with None
    Returns: bool
    """
    if array is None:
        return True
    if all(i is None for i in shape):
        return True

    array_shape = array.shape
    valid_dims = [i for i in shape if i is not None]
    n_for_check = len(valid_dims)
    return all(i == j for i, j in zip(array_shape[:n_for_check], valid_dims))


def getter_arr(obj, pi):
    """Get prop.
    """
    if "." in pi:
        pis = list(pi.split("."))
        pis.reverse()
        while len(pis):
            s = pis.pop()
            obj = getter_arr(obj, s)
        return obj
    elif "()" in pi:
        return getattr(obj, pi[:-2])()

    else:
        return getattr(obj, pi)


def temp_jump(mark=0, temp_device=None, old_device=None, temp=True, back=True):
    def f(func):
        return _temp_jump(func, mark=mark, temp_device=temp_device, old_device=old_device, temp=temp, back=back)

    return f


def temp_jump_cpu(mark=0, temp_device="cpu", old_device=None, temp=True, back=True):
    def f(func):
        return _temp_jump(func, mark=mark, temp_device=temp_device, old_device=old_device, temp=temp, back=back)

    return f


def _temp_jump(func, mark=0, temp_device="cpu", old_device=None, temp=True, back=True):
    """temp to cpu to calculate and re-back the init device data."""

    @wraps(func)
    def wrapper(*args, **kwargs):

        if temp_device is None:
            device = args[0].device
        else:
            device = torch.device(temp_device) if isinstance(temp_device, str) else temp_device

        if old_device is None:
            device2 = args[mark + 1].device if len(args) > 1 else list(kwargs.values())[0].device
        else:
            device2 = torch.device(old_device) if isinstance(old_device, str) else old_device

        if temp:
            args2 = [args[0]]
            for i in args[1:]:
                try:
                    args2.append(i.to(device=device, copy=False))
                except AttributeError:
                    args2.append(i)
            kwargs2 = {}
            for k, v in kwargs.items():
                try:
                    if isinstance(v, tuple):
                        kwargs2[k] = [i.to(device=device, copy=False) for i in v]
                    else:
                        kwargs2[k] = v.to(device=device, copy=False)
                except AttributeError:
                    kwargs2[k] = v

            result = func(*args2, **kwargs2)
        else:
            result = func(*args, **kwargs)

        if back:
            if isinstance(result, tuple):
                result2 = []
                for i in result:
                    try:
                        result2.append(i.to(device=device2, copy=False))
                    except AttributeError:
                        result2.append(i)
            else:
                try:
                    result2 = result.to(device=device2, copy=False)
                except AttributeError:
                    result2 = result
            return result2
        else:
            return result

    return wrapper


def check_device(mode: Module):
    device = _check_device(mode)
    return torch.device("cpu") if device is None else device


def _check_device(mode: Module):
    device = None
    for i in mode.children():
        if hasattr(i, "weight"):
            device = i.weight.device
            break
        elif hasattr(i, "bias"):
            device = i.bias.device
            break
        elif len(list(i.children())) > 0:
            device = check_device(i)
            if device is not None:
                break
    return device


def get_ptr(index):
    """Trans batch index to ptr"""
    return torch.ops.torch_sparse.ind2ptr(index, index[-1] + 1)


"""This file is not used, which is to fixed one bug in torch_geometrics"""


def separate(cls, batch: BaseData, idx: int, slice_dict: Any,
             inc_dict: Any = None, decrement: bool = True) -> BaseData:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    data = cls().stores_as(batch)

    # We iterate over each storage object and recursively separate all its
    # attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        key = batch_store._key
        if key is not None:
            attrs = slice_dict[key].keys()
        else:
            attrs = set(batch_store.keys())
            attrs = [attr for attr in slice_dict.keys() if attr in attrs]
        for attr in attrs:
            if key is not None:
                slices = slice_dict[key][attr]
                incs = inc_dict[key][attr] if decrement else None
            else:
                slices = slice_dict[attr]
                incs = inc_dict[attr] if decrement else None
            data_store[attr] = _separate(attr, batch_store[attr], idx, slices,
                                         incs, batch, batch_store, decrement)

        # The `num_nodes` attribute needs special treatment, as we cannot infer
        # the real number of nodes from the total number of nodes alone:
        if hasattr(batch_store, '_num_nodes'):
            data_store.num_nodes = batch_store._num_nodes[idx]

    return data


def _separate(
        key: str,
        value: Any,
        idx: int,
        slices: Any,
        incs: Any,
        batch: BaseData,
        store: BaseStorage,
        decrement: bool,
) -> Any:
    if isinstance(value, Tensor):
        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = value.narrow(cat_dim or 0, start, end - start)
        value = value.squeeze(0) if cat_dim is None else value
        if decrement and (incs.dim() > 1 or int(incs[idx]) != 0):
            value = value - incs[idx].to(value.device)
        return value

    elif isinstance(value, SparseTensor) and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        cat_dims = (cat_dim,) if isinstance(cat_dim, int) else cat_dim
        for i, dim in enumerate(cat_dims):
            start, end = int(slices[idx][i]), int(slices[idx + 1][i])
            value = value.narrow(dim, start, end - start)
        return value

    elif isinstance(value, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key: _separate(key, elem, idx, slices[key],
                           incs[key] if decrement else None, batch, store,
                           decrement)
            for key, elem in value.items()
        }

    elif (isinstance(value, Sequence) and isinstance(value[0], Sequence)
          and not isinstance(value[0], str) and len(value[0]) > 0
          and isinstance(value[0][0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of lists.
        return [elem[idx] for elem in value]

    elif (isinstance(value, Sequence) and not isinstance(value, str)
          and isinstance(value[0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of Tensors/SparseTensors.
        return [
            _separate(key, elem, idx, slices[i],
                      incs[i] if decrement else None, batch, store, decrement)
            for i, elem in enumerate(value)
        ]

    else:
        return value[idx]
