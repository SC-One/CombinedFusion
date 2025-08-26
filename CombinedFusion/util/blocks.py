import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicBlocks import ResidualConvUnit, ECABlock
from .BasicBlocks import SpaceEnhancedBlock, AxisTransformerBlock

class CombinedFusionBlock(nn.Module):
    """
    Combined fusion block with multiscale conv, space-enhanced conv, axial transformer, and ECA
    """
    def __init__(self,
                 features,
                 activation=nn.ReLU(False),
                 bn=False,
                 expand=False,
                 align_corners=True,
                 use_space=True,
                 use_transformer=False,
                 transformer_heads=2,
                 size=None):
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.expand = expand
        # output channels after upsample
        out_ch = features if not expand else features // 2

        # first residual
        self.res1 = ResidualConvUnit(features, activation, bn)

        # multiscale conv paths
        self.conv3 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(features, features, kernel_size=5, padding=2, bias=True)

        # space-enhanced conv branch
        self.use_space = use_space
        if use_space:
            self.space_branch = SpaceEnhancedBlock(channels=features)

        # axial transformer branch
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = AxisTransformerBlock(dim=features, num_heads=transformer_heads)

        # fuse
        n_streams = 2 + int(use_space) + int(use_transformer)
        self.fuse = nn.Conv2d(features * n_streams, features, kernel_size=1, bias=True)

        # ECA
        self.eca = ECABlock(features)

        # second residual
        self.res2 = ResidualConvUnit(features, activation, bn)

        # projection
        self.out_conv = nn.Conv2d(features, out_ch, kernel_size=1, bias=True)
        self.skip = nn.quantized.FloatFunctional()

    def forward(self, low_feat, high_feat=None, transformer_input=None, size=None):
        x = low_feat
        # fuse high-level skip
        if high_feat is not None:
            if high_feat.shape[-2:] != low_feat.shape[-2:]:
                high_feat = F.interpolate(high_feat,
                                           size=low_feat.shape[-2:],
                                           mode="bilinear",
                                           align_corners=self.align_corners)
            x = self.skip.add(x, high_feat)

        x = self.res1(x)
        streams = [self.conv3(x), self.conv5(x)]
        if self.use_space:
            streams.append(self.space_branch(x))
        if self.use_transformer and transformer_input is not None:
            t = transformer_input
            if t.shape[-2:] != x.shape[-2:]:
                t = F.interpolate(t,
                                  size=x.shape[-2:],
                                  mode="bilinear",
                                  align_corners=self.align_corners)
            streams.append(self.transformer(t))

        fused = torch.cat(streams, dim=1)
        fused = self.fuse(fused)
        fused = self.eca(fused)
        fused = self.skip.add(fused, x)
        out = self.res2(fused)

        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        out = F.interpolate(out, **modifier,
                             mode="bilinear",
                             align_corners=self.align_corners)
        return self.out_conv(out)


def _make_fusion_block(features, use_bn, size=None):
    """
    Factory method returning our CombinedFusionBlock with the same signature as before.
    """
    return CombinedFusionBlock(
        features=features,
        activation=nn.ReLU(False),
        bn=use_bn,
        expand=False,
        align_corners=True,
        use_space=True,
        use_transformer=False,
        transformer_heads=2,
        size=size
    )