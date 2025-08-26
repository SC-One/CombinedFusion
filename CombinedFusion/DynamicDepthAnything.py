import torch
from torch import nn
from .dpt import DepthAnythingV2
from .util.utilities import describe, freeze_model


class CombinedFusion(nn.Module):
    def __init__(self):
        super(CombinedFusion, self).__init__()
        # Note: we change head of DA-v2 codes and have one model config, instead different sizes.
        self.da1 = DepthAnythingV2(encoder='vits',
                                   features=64,
                                   out_channels=[48, 96, 192, 384],
                                   use_bn=False,
                                   use_clstoken=False)

    def verbose(self) -> str:
        return f'CombinedFusion verbosing\n{describe(self)}'

    # indoor: 20, outdoor:80 --- if you are not sure about max_depth, don't touch it.
    def forward(self, x, max_depth = 80): 
        result = self.da1(x, [2, 5, 8, 11], max_depth)
        return result
