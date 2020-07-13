# import torch
# MODELS = {
#     # Model name followed by the number of output classes.
#     "r2plus1d_34_32_ig65m": 359,
#     "r2plus1d_34_32_kinetics": 400,
#     "r2plus1d_34_8_ig65m": 487,
#     "r2plus1d_34_8_kinetics": 400,
# }
# model_R2_1 = torch.hub.load(
#     "moabitcoin/ig65m-pytorch",
#     "r2plus1d_34_8_ig65m",
#     num_classes=101,
#     pretrained=False,
# )
import numpy as np
import torch.nn as nn
import torch

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

import torch.nn as nn
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out
        out = _inner_forward(x)
        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        partial_bn (bool): Whether to freeze weight and bias of **all but the first** BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
                len(self.stage_blocks) - 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

#_____________________________________________ todo

class NonLocalModule(nn.Module):
    def __init__(self, in_channels=1024, nonlocal_type="gaussian", dim=3, embed=True, embed_dim=None, sub_sample=True, use_bn=True):
        super(NonLocalModule, self).__init__()

        assert nonlocal_type in ['gaussian', 'dot', 'concat']
        assert dim == 2 or dim == 3
        self.nonlocal_type = nonlocal_type
        self.embed = embed
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // 2
        self.sub_sample = sub_sample
        self.use_bn = use_bn

        if self.embed:
            if dim == 2:
                self.theta = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.phi = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.g = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            elif dim == 3:
                self.theta = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
                self.phi = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
                self.g = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        if self.nonlocal_type == 'gaussian':
            self.softmax = nn.Softmax(dim=2)
        elif self.nonlocal_type == 'concat':
            if dim == 2:
                self.concat_proj = nn.Sequential(nn.Conv2d(self.embed_dim * 2, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                                 nn.ReLU())
            elif dim == 3:
                self.concat_proj = nn.Sequential(nn.Conv3d(self.embed_dim * 2, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
                                                 nn.ReLU())

        if sub_sample:
            if dim == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
            elif dim == 3:
                self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.max_pool, self.g)
            self.phi = nn.Sequential(self.max_pool, self.phi)

        if dim == 2:
            self.W = nn.Conv2d(self.embed_dim, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        elif dim == 3:
            self.W = nn.Conv3d(self.embed_dim, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        if use_bn:
            if dim == 2:
                self.bn = nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.9, affine=True)
            elif dim == 3:
                self.bn = nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.9, affine=True)
            self.W = nn.Sequential(self.W, self.bn)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 0)


    def forward(self, input):
        if self.embed:
            theta = self.theta(input)
            phi = self.phi(input)
            g = self.g(input)
        else:
            theta = input
            phi = input
            g = input

        if self.nonlocal_type in ['gaussian', 'dot']:
            # reshape [BxC'xTxHxW] to [BxC'x(T)HW]
            theta = theta.reshape(theta.shape[:2] + (-1,))
            phi = phi.reshape(theta.shape[:2] + (-1,))
            g = g.reshape(theta.shape[:2] + (-1,))
            theta_phi = torch.matmul(theta.transpose(1, 2), phi)
            if self.nonlocal_type == 'gaussian':
                p = self.softmax(theta_phi)
            elif self.nonlocal_type == 'dot':
                N = theta_phi.size(-1)
                p = theta_phi / N
        elif self.non_local_type == 'concat':
            # reshape [BxC'xTxHxW] to [BxC'x(T)HWx1]
            theta = theta.reshape(theta.shape[:2] + (-1,1))
            # reshape [BxC'xTxHxW] to [BxC'x1x(T)HW]
            phi = phi.reshape(theta.shape[:2] + (1,-1))
            theta_x = theta.repeat(1, 1, 1, phi.size(3))
            phi_x = phi.repeat(1, 1, theta.size(2), 1)
            theta_phi = torch.cat([theta_x, phi_x], dim=1)
            theta_phi = self.concat_proj(theta_phi)
            theta_phi = theta_phi.squeeze()
            N = theta_phi.size(-1)
            p = theta_phi / N
        else:
            NotImplementedError

        # BxC'xddd , Bxdxddd => BxC'xd
        y = torch.matmul(g, p.transpose(1, 2))
        y = y.reshape(y.shape[:2] + input.shape[2:])
        z = self.W(y) + input

        return z


def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias=False)



#_--------------------------------------------------
from resnet_r3d_utils import *


class BasicBlock(nn.Module):
    def __init__(self,
                 input_filters,
                 num_filters,
                 base_filters,
                 down_sampling=False,
                 down_sampling_temporal=None,
                 block_type='3d',
                 is_real_3d=True,
                 group=1,
                 with_bn=True):


        super(BasicBlock, self).__init__()
        self.num_filters = num_filters
        self.base_filters = base_filters
        self.input_filters = input_filters
        self.with_bn = with_bn
        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias

        if block_type == '2.5d':
            assert is_real_3d
        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                self.down_sampling_stride = [2, 2, 2]
            else:
                self.down_sampling_stride = [1, 2, 2]
        else:
            self.down_sampling_stride = [1, 1, 1]

        self.down_sampling = down_sampling

        self.relu = nn.ReLU()
        self.conv1 = add_conv3d(input_filters, num_filters,
                                kernel=[3, 3, 3] if is_real_3d else [1, 3, 3],
                                stride=self.down_sampling_stride,
                                pad=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                block_type=block_type, with_bn=self.with_bn)
        if self.with_bn:
            self.bn1 = add_bn(num_filters)
        self.conv2 = add_conv3d(num_filters, num_filters,
                                kernel=[3, 3, 3] if is_real_3d else [1, 3, 3],
                                stride=[1, 1, 1],
                                pad=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                block_type=block_type, with_bn=self.with_bn)
        if self.with_bn:
            self.bn2 = add_bn(num_filters)
        if num_filters != input_filters or down_sampling:
            self.conv3 = conv3d(input_filters, num_filters, kernel=[1, 1, 1],
                                stride=self.down_sampling_stride, pad=[0, 0, 0])
            if self.with_bn:
                self.bn3 = nn.BatchNorm3d(num_filters, eps=1e-3)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.with_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_bn:
            out = self.bn2(out)

        if self.down_sampling or self.num_filters != self.input_filters:
            identity = self.conv3(identity)
            if self.with_bn:
                identity = self.bn3(identity)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self,
                 input_filters,
                 num_filters,
                 base_filters,
                 down_sampling=False,
                 down_sampling_temporal=None,
                 block_type='3d',
                 is_real_3d=True,
                 group=1,
                 with_bn=True):

        super(Bottleneck, self).__init__()
        self.num_filters = num_filters
        self.base_filters = base_filters
        self.input_filters = input_filters
        self.with_bn = with_bn
        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias

        if block_type == '2.5d':
            assert is_real_3d
        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                self.down_sampling_stride = [2, 2, 2]
            else:
                self.down_sampling_stride = [1, 2, 2]
        else:
            self.down_sampling_stride = [1, 1, 1]

        self.down_sampling = down_sampling
        self.relu = nn.ReLU()

        self.conv0 = add_conv3d(input_filters, base_filters, kernel=[
            1, 1, 1], stride=[1, 1, 1], pad=[0, 0, 0], with_bn=self.with_bn)
        if self.with_bn:
            self.bn0 = add_bn(base_filters)

        self.conv1 = add_conv3d(base_filters, base_filters,
                                kernel=[3, 3, 3] if is_real_3d else [1, 3, 3],
                                stride=self.down_sampling_stride,
                                pad=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                block_type=block_type, with_bn=self.with_bn)
        if self.with_bn:
            self.bn1 = add_bn(base_filters)

        self.conv2 = add_conv3d(base_filters, num_filters, kernel=[
            1, 1, 1], pad=[0, 0, 0], stride=[1, 1, 1], with_bn=self.with_bn)

        if self.with_bn:
            self.bn2 = add_bn(num_filters)

        if num_filters != input_filters or down_sampling:
            self.conv3 = conv3d(input_filters, num_filters, kernel=[1, 1, 1],
                                stride=self.down_sampling_stride, pad=[0, 0, 0])
            if self.with_bn:
                self.bn3 = nn.BatchNorm3d(num_filters, eps=1e-3)

    def forward(self, x):
        identity = x
        if self.with_bn:
            out = self.relu(self.bn0(self.conv0(x)))
            out = self.relu(self.bn1(self.conv1(out)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.relu(self.conv0(x))
            out = self.relu(self.conv1(out))
            out = self.conv2(out)

        if self.down_sampling or self.num_filters != self.input_filters:
            identity = self.conv3(identity)
            if self.with_bn:
                identity = self.bn3(identity)

        out += identity
        out = self.relu(out)
        return out


def make_plain_res_layer(block, num_blocks, in_filters, num_filters, base_filters,
                         block_type='3d', down_sampling=False, down_sampling_temporal=None,
                         is_real_3d=True, with_bn=True):
    layers = []
    layers.append(block(in_filters, num_filters, base_filters, down_sampling=down_sampling,
                        down_sampling_temporal=down_sampling_temporal, block_type=block_type,
                        is_real_3d=is_real_3d, with_bn=with_bn))
    for i in range(num_blocks - 1):
        layers.append(block(num_filters, num_filters, base_filters,
                            block_type=block_type, is_real_3d=is_real_3d, with_bn=with_bn))
    return module_list(layers)


BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}
SHALLOW_FILTER_CONFIG = [
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512]
]
DEEP_FILTER_CONFIG = [
    [256, 64],
    [512, 128],
    [1024, 256],
    [2048, 512]
]


class ResNet_R3D(nn.Module):

    def __init__(self,
                 pretrained=None,
                 num_input_channels=3,
                 depth=34,
                 block_type='2.5d',
                 channel_multiplier=1.0,
                 bottleneck_multiplier=1.0,
                 conv1_kernel_t=3,
                 conv1_stride_t=1,
                 use_pool1=False,
                 bn_eval=True,
                 bn_frozen=True,
                 with_bn=True):
        #         parameter initialization
        super(ResNet_R3D, self).__init__()
        self.pretrained = pretrained
        self.num_input_channels = num_input_channels
        self.depth = depth
        self.block_type = block_type
        self.channel_multiplier = channel_multiplier
        self.bottleneck_multiplier = bottleneck_multiplier
        self.conv1_kernel_t = conv1_kernel_t
        self.conv1_stride_t = conv1_stride_t
        self.use_pool1 = use_pool1
        self.relu = nn.ReLU()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_bn = with_bn
        global comp_count, comp_idx
        comp_idx = 0
        comp_count = 0

        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias

        #         stem block
        if self.block_type in ['2.5d', '2.5d-sep']:
            self.conv1_s = conv3d(self.num_input_channels, 45, [
                1, 7, 7], [1, 2, 2], [0, 3, 3])
            if self.with_bn:
                self.bn1_s = nn.BatchNorm3d(45, eps=1e-3)
            self.conv1_t = conv3d(45, 64, [self.conv1_kernel_t, 1, 1], [self.conv1_stride_t, 1, 1],
                                  [(self.conv1_kernel_t - 1) // 2, 0, 0])
            if self.with_bn:
                self.bn1_t = nn.BatchNorm3d(64, eps=1e-3)
        else:
            self.conv1 = conv3d(self.num_input_channels, 64, [self.conv1_kernel_t, 7, 7],
                                [self.conv1_stride_t, 2, 2], [(self.conv1_kernel_t - 1) // 2, 3, 3])
            if self.with_bn:
                self.bn1 = nn.BatchNorm3d(64, eps=1e-3)

        if self.use_pool1:
            self.pool1 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[
                1, 2, 2], padding=[0, 1, 1])

        self.stage_blocks = BLOCK_CONFIG[self.depth]
        if self.depth <= 18 or self.depth == 34:
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        if self.depth <= 34:
            self.filter_config = SHALLOW_FILTER_CONFIG
        else:
            self.filter_config = DEEP_FILTER_CONFIG
        self.filter_config = np.multiply(
            self.filter_config, self.channel_multiplier).astype(np.int)

        layer1 = make_plain_res_layer(self.block, self.stage_blocks[0],
                                      64, self.filter_config[0][0],
                                      int(self.filter_config[0][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type,
                                      with_bn=self.with_bn)
        self.add_module('layer1', layer1)
        layer2 = make_plain_res_layer(self.block, self.stage_blocks[1],
                                      self.filter_config[0][0], self.filter_config[1][0],
                                      int(self.filter_config[1][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type, down_sampling=True,
                                      with_bn=self.with_bn)
        self.add_module('layer2', layer2)
        layer3 = make_plain_res_layer(self.block, self.stage_blocks[2],
                                      self.filter_config[1][0], self.filter_config[2][0],
                                      int(self.filter_config[2][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type, down_sampling=True,
                                      with_bn=self.with_bn)
        self.add_module('layer3', layer3)
        layer4 = make_plain_res_layer(self.block, self.stage_blocks[3],
                                      self.filter_config[2][0], self.filter_config[3][0],
                                      int(self.filter_config[3][1]
                                          * self.bottleneck_multiplier),
                                      block_type=self.block_type, down_sampling=True,
                                      with_bn=self.with_bn)
        self.add_module('layer4', layer4)
        self.res_layers = ['layer1', 'layer2', 'layer3', 'layer4']

    def forward(self, x):
        if self.block_type in ['2.5d', '2.5d-sep']:
            if self.with_bn:
                x = self.relu(self.bn1_s(self.conv1_s(x)))
                x = self.relu(self.bn1_t(self.conv1_t(x)))
            else:
                x = self.relu(self.conv1_s(x))
                x = self.relu(self.conv1_t(x))
        else:
            if self.with_bn:
                x = self.relu(self.bn1(self.conv1(x)))
            else:
                x = self.relu(self.conv1(x))

        if self.use_pool1:
            x = self.pool1(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def train(self, mode=True):
        super(ResNet_R3D, self).train(mode)
        if self.bn_eval and self.with_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

# if spatial_type == 'avg':
#     self.pool_func = nn.AdaptiveAvgPool3d(self.pool_size)
# if self.spatial_type == 'max':
#     self.pool_func = nn.AdaptiveMaxPool3d(self.pool_size)


class ClsHead(nn.Module):
    """Simplest classification head"""

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=101,
                 init_std=0.01,
                 fcn_testing=False):

        super(ClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing
        self.num_classes = num_classes

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size))

        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.new_cls = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if not self.fcn_testing:
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = x.view(x.size(0), -1)

            cls_score = self.fc_cls(x)
            return cls_score
        else:
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1,1,0).cuda()
                self.new_cls.load_state_dict({'weight': self.fc_cls.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                              'bias': self.fc_cls.bias})
            class_map = self.new_cls(x)
            return class_map

class model_r3d(nn.Module):
    def __init__(self,num_classes,depth=34,dropout_ratio=0.8):
        super(model_r3d, self).__init__()
        self.m1  = ResNet_R3D(depth=depth)
        self.m1.init_weights()
        self.pool_func = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc_cls = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pool_func(self.m1(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc_cls(x)

if __name__ == '__main__':
    model = model_r3d(101)
    a =torch.ones(1,3,10,112,112)
    print(model(a).shape)