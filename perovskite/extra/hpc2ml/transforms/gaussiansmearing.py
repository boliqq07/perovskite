import warnings

import torch
from torch_geometric.transforms import BaseTransform


class GaussianSmearing(BaseTransform):
    """Smear the radius shape (num_node,1) to shape (num_node, num_edge_gaussians)."""

    def __init__(self, start=0.0, stop=5.0, num_edge_gaussians=50):
        super(GaussianSmearing, self).__init__()
        self.offset = torch.linspace(start, stop, num_edge_gaussians)
        self.coef = -0.5 / (self.offset[1] - self.offset[0]).item() ** 2

    def __call__(self, data):
        dist = data.edge_weight
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        if hasattr(data, "edge_attr") and data.edge_attr.shape[1] != 1:
            warnings.warn("The old edge_attr is covered by smearing edge_weight", UserWarning)
        data.edge_attr = torch.exp(self.coef * torch.pow(dist, 2))

        return data
