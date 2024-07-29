import math
import torch
import torch.nn as nn

from autoencoder.utils import activation_factory

# From GPM
# https://github.com/White-Link/gpm/blob/master/gpm/networks/mlp.py


class MLP(nn.Module):
    """
    MLP architecture. Can optionnally take as input two tensors that are then concatenated.
    """

    def __init__(self, in_size, out_size, width, depth, activation):
        super().__init__()

        if depth >= 0:
            if isinstance(in_size, int):
                in_dim = in_size
            else:
                in_dim = math.prod(in_size)
            if isinstance(out_size, int):
                out_dim = out_size
            else:
                out_dim = math.prod(out_size)

            if isinstance(activation, list):
                activation_fns = list(map(activation_factory, activation))
            else:
                activation_fns = [activation_factory(activation)] * depth
            hidden_width = width if depth > 0 else out_dim
            layers = [
                nn.Linear(in_dim, hidden_width),
                *[
                    nn.Sequential(
                        activation_fns[i],
                        nn.Linear(width, width if i < depth - 1 else out_dim)
                    )
                    for i in range(depth)
                ],
            ]
            if not isinstance(out_size, int):
                layers.append(torch.nn.Unflatten(1, tuple(out_size)))

            self.mlp = nn.Sequential(*layers)

        else:
            assert in_size == out_size
            self.mlp = nn.Identity()