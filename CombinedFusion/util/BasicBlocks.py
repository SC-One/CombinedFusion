import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
 
def _make_scratch(in_shape, out_shape, groups=1, expand=False): 
    """ 
    Creates a set of convolutional layers to project incoming feature maps to a common dimension. 
    """ 
    scratch = nn.Module() 
 
    out_shape1 = out_shape 
    out_shape2 = out_shape 
    out_shape3 = out_shape 
    if len(in_shape) >= 4: 
        out_shape4 = out_shape 
 
    if expand: 
        out_shape1 = out_shape 
        out_shape2 = out_shape * 2 
        out_shape3 = out_shape * 4 
        if len(in_shape) >= 4: 
            out_shape4 = out_shape * 8 
 
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups) 
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups) 
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups) 
    if len(in_shape) >= 4: 
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups) 
 
    return scratch 
 
 
class ResidualConvUnit(nn.Module): 
    """Residual convolution unit with optional batchnorm and skip connection.""" 
    def __init__(self, features, activation, bn=False): 
        super().__init__() 
        self.bn = bn 
        self.activation = activation 
        self.groups = 1 
 
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups) 
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups) 
 
        if self.bn: 
            self.bn1 = nn.BatchNorm2d(features) 
            self.bn2 = nn.BatchNorm2d(features) 
 
        self.skip_add = nn.quantized.FloatFunctional() 
 
    def forward(self, x): 
        out = self.activation(x) 
        out = self.conv1(out) 
        if self.bn: 
            out = self.bn1(out) 
 
        out = self.activation(out) 
        out = self.conv2(out) 
        if self.bn: 
            out = self.bn2(out) 
 
        return self.skip_add.add(out, x) 
 
 
class ECABlock(nn.Module): 
    """Efficient Channel Attention (ECA).""" 
    def __init__(self, channels, k_size=3): 
        super().__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        # 1D conv for local cross-channel interaction 
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                              padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid() 
 
    def forward(self, x): 
        # x: B×C×H×W 
        y = self.avg_pool(x)            # B×C×1×1 
        y = y.squeeze(-1).transpose(-1, -2)  # B×1×C 
        y = self.conv(y)                # B×1×C 
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # B×C×1×1 
        return x * y.expand_as(x) 
 
 
class AxisWindowBlock(nn.Module): 
    """Axial self-attention along a single spatial dimension.""" 
    def __init__(self, dim, num_heads): 
        super().__init__() 
        self.norm1 = nn.LayerNorm(dim) 
        self.attn = nn.MultiheadAttention(dim, num_heads) 
        self.mlp = nn.Sequential( 
            nn.LayerNorm(dim), 
            nn.Linear(dim, dim * 4), 
            nn.GELU(), 
            nn.Linear(dim * 4, dim) 
        ) 
 
    def forward(self, x, axis=2): 
        # x: B×C×H×W, axis=2 for height, 3 for width 
        B, C, H, W = x.shape 
        if axis == 2: 
            x_perm = x.permute(2, 0, 3, 1).reshape(H, B * W, C) 
        else: 
            x_perm = x.permute(3, 0, 2, 1).reshape(W, B * H, C) 
 
        x_norm = self.norm1(x_perm) 
        attn_out, _ = self.attn(x_norm, x_norm, x_norm) 
        x2 = x_perm + attn_out 
        mlp_out = self.mlp(x2) 
        x3 = x2 + mlp_out 
 
        if axis == 2: 
            x3 = x3.reshape(H, B, W, C).permute(1, 3, 0, 2) 
        else: 
            x3 = x3.reshape(W, B, H, C).permute(1, 3, 2, 0) 
        return x3 
 
 
class AxisTransformerBlock(nn.Module): 
    """Combines axial attention over height and width axes.""" 
    def __init__(self, dim, num_heads=4): 
        super().__init__() 
        self.awb_h = AxisWindowBlock(dim, num_heads) 
        self.awb_w = AxisWindowBlock(dim, num_heads) 
 
    def forward(self, x): 
        x = self.awb_h(x, axis=2) 
        x = self.awb_w(x, axis=3) 
        return x 
 
 
class SpaceEnhancedBlock(nn.Module): 
    """Convolutional branch with spatial and channel attention.""" 
    def __init__(self, channels, groups=8): 
        super().__init__() 
        mid = channels 
        self.gconv1 = nn.Conv2d(channels, mid, kernel_size=1, groups=groups) 
        self.dwconv = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid) 
        self.gconv2 = nn.Conv2d(mid, channels, kernel_size=1, groups=groups) 
        self.elu = nn.ELU(inplace=True) 
        self.eca = ECABlock(channels) 
        self.spa = nn.Sequential( 
            nn.Conv2d(channels, channels // 4, kernel_size=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels // 4, 1, kernel_size=1), 
            nn.Sigmoid() 
        ) 
 
    def forward(self, x): 
        identity = x 
        out = self.elu(self.gconv1(x)) 
        out = self.elu(self.dwconv(out)) 
        out = self.elu(self.gconv2(out)) 
        out = identity + out 
        out = self.eca(out) 
        sa = self.spa(out) 
        return out * sa 
