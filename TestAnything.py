
import torch
import torch.nn as nn

from dc_ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization,
    timestep_embedding,
)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

if __name__ == '__main__':
    out = nn.Sequential(
                normalization(32),
                nn.SiLU(),
                zero_module(conv_nd(2, 32, 3, 3, padding=1)),
                )
    out