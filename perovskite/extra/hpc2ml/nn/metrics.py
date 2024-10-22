import torch
from torch.nn import functional as F


def mlm(y_preds: tuple, batch_ys: tuple):
    ls = []
    for i, j in zip(y_preds, batch_ys):
        ls.append(F.l1_loss(i, j, reduction="mean"))
    return sum(ls)


def mlm_alpha(y_preds: tuple, batch_ys: tuple, alphas=None):
    if alphas is None:
        alphas = torch.ones(len(y_preds))
    ls = []
    for i, j, k in zip(y_preds, batch_ys, alphas):
        ls.append(k * F.l1_loss(i, j, reduction="mean"))
    return sum(ls)
