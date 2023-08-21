from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
import warnings
from itertools import repeat
import collections.abc
import numbers
from einops import rearrange


__all__ = [
    'Mlp_Origin', 'Mlp_Dw', 'FeedForward', 'payattn_mlp', 'sparse_MLP',
    'LayerNorm', 'LayerNorm_restormer',
    'trunc_normal_', '_no_grad_trunc_normal_',
    'drop_path', 'DropPath', '_ntuple',
    'xavier_init', 'normal_init', 'trunc_normal_init', 'constant_init'
 ]

""" ---------------------------------- MLP ---------------------------------- """
class Mlp_Origin(nn.Module):
    """ Original """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_Dw(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features,hidden_features,3,1,1,groups=hidden_features)#DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        # self.attn = BiAttn(out_features)
        # self.attn = eca(out_features)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # x=self.attn(x)

        return x


class FeedForward(nn.Module):
    """https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in_ = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_ = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out_ = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.attn = BiAttn(dim)
        # self.attn = eca(dim)

    def forward(self, x):
        x = self.project_in_(x)
        x1, x2 = self.dwconv_(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2 #self.ca(x2)
        x = self.project_out_(x)
        x = self.attn(x)
        return x


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = LayerNorm(in_channels, data_format='channels_first')
        self.global_reduce = nn.Conv2d(in_channels, reduce_channels, 1) #Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Conv2d(in_channels, reduce_channels, 1)
        self.act_fn = act_fn()
        self.channel_select = nn.Conv2d(reduce_channels, in_channels, 1)
        self.spatial_select = nn.Conv2d(reduce_channels * 2, 1, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        b,c,h,w=x.shape
        x = self.norm(x)
        x_global = F.adaptive_avg_pool2d(x, 1) #x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(b, -1, x.shape[2], x.shape[3])], dim=1)) #.expand(-1, x.shape[1], -1)
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn

class payattn_mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(payattn_mlp, self).__init__()
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)  # DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features//2, out_features, 1)
        self.spatial_proj = nn.Conv2d(hidden_features // 2, hidden_features // 2, kernel_size=1)
        self.norm = LayerNorm(hidden_features // 2 )
        # nn.init.constant_(self.spatial_proj.bias, 1.0)
        self.attn = BiAttn(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        u, v = x.chunk(2, dim=1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        out = self.fc2(out)
        out = self.attn(out)
        return out

class sparse_MLP(nn.Module):
    def __init__(self, W, H, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.Identity()#nn.BatchNorm2d(channels)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proh_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)
        self.attn = BiAttn(channels)

    def forward(self, x):
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        x = self.attn(x)
        return x


""" ---------------------------------- Utils ---------------------------------- """

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        # >>> w = torch.empty(3, 5)
        # >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



"""  weight init  """
def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore



""" LayerNorm """
class LayerNorm_restormer(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasWith'):
        super(LayerNorm_restormer, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type =='BiasWith':
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


if __name__ == '__main__':
    x=torch.randn(1,64,32,32)
    y=F.adaptive_avg_pool2d(x, 1)
    att=BiAttn(64)
    z=att(x)
    print(z.shape)

    zz=torch.randn(1,64,1,1)
    print((zz.expand(1, -1, x.shape[2], x.shape[3])).shape)
