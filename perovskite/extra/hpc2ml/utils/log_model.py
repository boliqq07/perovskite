# -*- coding: utf-8 -*-

# @Time    : 2021/8/1 18:07
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import copy
from typing import Union, List

import numpy as np
import torch
from torch.nn import Module
from torch_geometric.data import InMemoryDataset, DataLoader, Data

from hpc2ml.data.dataset import SimpleDataset


def print_log_dataset(dataset: Union[InMemoryDataset, SimpleDataset, List[Data], Data]):
    nfeat_node = -1
    nfeat_edge = -1
    nfeat_state = -1

    if isinstance(dataset, List):
        dataset = dataset[0]
    if isinstance(dataset, Data):
        data = dataset
    else:
        data = dataset.data

    assert isinstance(data, Data)

    print("\n数据信息:\n###############")

    try:
        print("图（化合物）数={},".format(max(data.idx) + 1))
    except AttributeError:
        pass
    try:
        nfeat_node = data.x.shape
        print("原子（节点）总个数: {}, 原子特征数: {}".format(nfeat_node[0], nfeat_node[1]))
    except AttributeError:
        pass

    try:
        nfeat_edge = data.edge_attr.shape

        print("键（连接）连接个数: {}, 键特征数: {}".format(nfeat_edge[0], nfeat_edge[1]))
    except AttributeError:
        pass

    try:
        nfeat_state = data.state_attr.shape

        print("状态数: {}, 状态特征数 {}".format(nfeat_state[0], nfeat_state[1]))
    except AttributeError:
        pass

    print("\n建议参数如下(若后处理，以处理后为准):")
    if nfeat_node != -1:
        print("nfeat_node={},".format(nfeat_node[1]))
    if nfeat_edge != -1:
        print("nfeat_edge={},".format(nfeat_edge[1]))
    if nfeat_state != -1:
        print("nfeat_state={},".format(nfeat_state[1]))


def print_log_dataloader(dataloader: DataLoader, print_index_of_sample=False):
    dataloader0 = copy.copy(dataloader)
    names = ['x', 'edge_attr', 'energy', 'pos', 'batch', 'ptr', 'z', 'idx', 'state_attr', 'adj_t', 'edge_weight',
             "num_graphs"]
    for data in dataloader0:
        shapes = []
        for i in names:
            shapes.append(getattr(data, i, None))

        print("\n每批信息（示例第0批）:\n###############")

        if shapes[-1] is not None:
            print("图（化合物）数={},".format(np.array(shapes[-1])))
        if shapes[0] is not None:
            print("节点（原子）数={}, 原子特征数={}".format(shapes[0].shape[0], shapes[0].shape[1]))
        if shapes[1] is not None:
            print("键（连接）数={}, 键特征数={}".format(shapes[1].shape[0], shapes[1].shape[1]))
        if shapes[8] is not None:
            print("状态数={}, 状态特征数={}".format(shapes[8].shape[0], shapes[8].shape[1]))

        if print_index_of_sample:
            if shapes[7] is not None:
                print("样本序号={},".format(np.array(shapes[7])))

        print("\n建议参数如下:")
        if shapes[0] is not None:
            print("nfeat_node={},".format(shapes[0].shape[1]))
        if shapes[1] is not None:
            print("nfeat_edge={},".format(shapes[1].shape[1]))
            print("num_edge_gaussians={},".format(shapes[1].shape[1]))
        if shapes[8] is not None:
            print("nfeat_state={},".format(shapes[8].shape[1]))

        break


def make_dot_(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
    # 画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    # 蓝色节点表示有梯度计算的变量Variables;
    # 橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.
    """
    Produces Graphviz representation of PyTorch autograd graph.

    First install graphviz:
        https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224  (windows)

        yum install graphviz (centos)

        apt-get install graphviz (ubuntu)

    Second:
        pip install graphviz
        pip install torchviz

    use:
        >>> vis_graph = make_dot_(y_pred, params=dict(list(model.named_parameters())))
        >>> vis_graph.render(format="pdf")

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:

     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
            params=dict(model.named_parameters()
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    from torchviz import make_dot
    return make_dot(var, params=params, show_attrs=show_attrs, show_saved=show_saved, max_attr_chars=max_attr_chars)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def record(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n  # (if pytorch not support +=)
        self.count += n
        self.avg = self.sum / self.count

    def text(self):
        return ' {loss.name} {loss.val:.4f} ({loss.avg:.4f}) '.format(loss=self)


class AverageMeterTotal(dict):

    def __init__(self, *names):
        super(AverageMeterTotal, self).__init__()
        for i in names:
            self.update({i: AverageMeter(i)})

    def reset(self):
        [v.reset() for k, v in self.items()]

    def text(self):
        return "".join([vi.text() for vi in self.values()])

    def record(self, name, val, n=1):
        self[name].record(val, n)


def get_layers_with_weight(model: Module, add_name=""):
    """Get layers with weight. use get_target_layer function ranther than this is suggested."""
    name_weight_module = {}
    for name, layer in model._modules.items():
        name = "{}->{}".format(add_name, name)
        # if isinstance(layer,Module):
        if len(layer._modules) > 0:
            name_weight_module.update(get_layers_with_weight(layer, name))
        else:
            if hasattr(layer, "weight"):
                name_weight_module[name] = layer
            else:
                pass
    return name_weight_module


def get_target_layer(model, target_layer) -> dict:
    """which layer to hook.  run model.named_modules() to get the name of layer.
    """
    mod = {}
    if isinstance(target_layer, (list, tuple)) and len(target_layer) > 0:

        for i in target_layer:
            for name, module in model.named_modules():
                if name == i or module is i:
                    mod.update({name: module})
                    break
    elif isinstance(target_layer, str):

        if target_layer == "all":
            # all layer
            for name, module in model.named_modules():
                mod.update({name: module})
        elif target_layer == "top":
            # top layer
            mod.update(model._modules)
        elif target_layer == "weight":
            # weight layer
            for name, module in model.named_modules():
                if hasattr(module, "weight"):
                    mod[name] = module
        elif target_layer == "my_weight":
            # weight layer
            mmdict = get_layers_with_weight(model)
            mod.update(mmdict)

    else:

        mmdict = get_layers_with_weight(model)
        mod.update(mmdict)

    return mod


class LogModule:
    """Get message of module."""

    def __init__(self, model, target_layer="weight"):
        self._stats_log = []
        self.stats_log_i = {}
        self.stats_log_format = {}

        self.layer_dict = get_target_layer(model, target_layer)

    def _get_weight(self):
        name_weight_module = {}
        for name, layer in self.layer_dict.items():
            if hasattr(layer, "weight"):
                v = layer.weight.detach().cpu().numpy().copy()
                name_weight_module[name] = np.array([np.mean(v, axis=0), np.std(v, axis=0)])
        return name_weight_module

    def record(self, append=True):
        stats_log_i = self._get_weight()
        if append:
            self._stats_log.append(stats_log_i)
            self.stats_log_i = self._stats_log[-1]
        else:
            self.stats_log_i = stats_log_i

    def stats_single(self):
        return self.stats_log_i

    def stats_loop(self):

        result = {}

        for i in self._stats_log[0].keys():
            nps = []
            for n, v in enumerate(self._stats_log):
                nps.append(v[i])

            result[i] = np.array(nps)

        self.stats_log_format = result

        return result


class HookGradientLayer:
    """This is used for one sample to check which is the import message for this target."""

    def __init__(self, layer_name, layer):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.layer_name = layer_name
        self._stats_log = []
        self.stats_log_i = {}
        self.stats_log_format = {}

        self.forward_hook = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_full_backward_hook(self.hook_fn_grad)
        # self.backward_hook = layer.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def _get_weight(self):
        try:
            activations = self.activations
            gradients = self.gradients
            v = (gradients * activations)
            v = v.detach().cpu().numpy()
            if v.ndim >= 2:
                return {self.layer_name: np.array([np.mean(v, axis=0), np.std(v, axis=0)])}
            else:
                return {self.layer_name: np.array([v, np.zeros_like(v)])}
        except:
            return {}

    def record(self, append=True):
        stats_log_i = self._get_weight()
        if append:
            self._stats_log.append(stats_log_i)
            self.stats_log_i = self._stats_log[-1]
        else:
            self.stats_log_i = stats_log_i

    def stats_single(self):
        return self.stats_log_i

    def stats_loop(self):
        result = {}
        for i in self._stats_log[0].keys():
            nps = []
            for n, v in enumerate(self._stats_log):
                nps.append(v[i])

            result[i] = np.array(nps)

        self.stats_log_format = result

        return result


class HookGradientModule():
    def __init__(self, model, target_layer="all"):
        self.stats_log_format = []
        self.stats_log_i = {}

        self.layer_dict = get_target_layer(model, target_layer)

        self.svls = []
        for k, v in self.layer_dict.items():
            svl = HookGradientLayer(k, v)
            self.svls.append(svl)

    def apply(self, func, **kwargs):
        result = []
        for i in self.svls:
            ri = getattr(i, func)(**kwargs)
            result.append(ri)
        return result

    def record(self, append=True):
        self.apply("record", append=append)

    def stats_single(self):
        self.stats_log_i = {}
        results = self.apply("stats_single")
        [self.stats_log_i.update(i) for i in results]
        return self.stats_log_i

    def stats_loop(self):
        self.stats_log_format = {}
        results = self.apply("stats_loop")
        [self.stats_log_format.update(i) for i in results]
        return self.stats_log_format


# def for_hook(module, input, output):
#     print(module)
#     for val in input:
#         print("input val:", val)
#     for out_val in output:
#         print("output val:", out_val)

def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print(
        "Max Memory Allocated:",
        torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))
