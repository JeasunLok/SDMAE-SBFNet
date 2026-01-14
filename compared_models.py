import numpy as np
import torch
import os
import torch.nn.functional as F
import math
from functools import partial
from torch import nn, einsum
from einops import rearrange, reduce, repeat, parse_shape
import scipy as sp

# ====================== UNet ====================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)# //为整数除法

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='trunc'), 
                diffX - torch.div(diffX, 2, rounding_mode='trunc'),
                torch.div(diffY, 2, rounding_mode='trunc'),
                diffY - torch.div(diffY, 2, rounding_mode='trunc')])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ====================== DeepLabV3P ====================

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1    = norm_layer(planes)
        
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2    = norm_layer(planes)
        
        self.conv3  = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3    = norm_layer(planes * 4)
        
        self.relu   = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation   = dilation
        self.stride     = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, n_channels=3, n_classes=1000, dilated=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128
        self.n_channels= n_channels
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.bn1        = norm_layer(self.inplanes)
        self.relu       = nn.ReLU(inplace=True)
        
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1     = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2     = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                            dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
            
        self.avgpool    = nn.AvgPool2d(7, stride=1)
        self.fc         = nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(n_channels=3, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_channels=n_channels, **kwargs)
    return model

class Resnet(nn.Module):
    def __init__(self, n_channels=3, dilate_scale=16):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(n_channels=n_channels)

        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1[0]
        self.bn1 = model.conv1[1]
        self.relu1 = model.conv1[2]
        self.conv2 = model.conv1[3]
        self.bn2 = model.conv1[4]
        self.relu2 = model.conv1[5]
        self.conv3 = model.conv1[6]
        self.bn3 = model.bn1
        self.relu3 = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x

class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()

		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)

		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLabV3P(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DeepLabV3P, self).__init__()
        downsample_factor = 16
        self.backbone = Resnet(n_channels=n_channels, dilate_scale=downsample_factor)
        in_channels = 2048
        low_level_channels = 1024

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, n_classes, 1, stride=1)
        self.bn0 = nn.BatchNorm2d(2048)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        low_level_features, x = self.backbone(x)
        x = self.bn0(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# ====================== PSPNet ====================

class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        
        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(PSPNet, self).__init__()
        downsample_factor = 16
        aux_branch = False
        norm_layer = nn.BatchNorm2d
        self.backbone = Resnet(n_channels, downsample_factor)
        aux_channel = 1024
        out_channel = 2048
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel//4, n_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, n_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
                    
# ====================== SegFormer ====================

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(math.sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer(nn.Module):
    def __init__(
        self,
        n_channels = 3,
        n_classes = 8,
        dims = (64, 128, 320, 512),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = (3, 8, 27, 3),
        decoder_dim = 256
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = n_channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        # self.to_segmentation = nn.Sequential(
        #     nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
        #     nn.Conv2d(decoder_dim, n_classes, 1),
        # )
        
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, n_classes, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 4倍上采样
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        return self.to_segmentation(fused)

# SETR 
def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    """ 
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

class TransConfig(object):
    
    def __init__(
        self,
        patch_size,
        in_channels,
        out_channels,
        sample_rate=4,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=6,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):  
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class TransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
       

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
      
class TransEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        input_shape = input_ids.size()
    
        seq_length = input_shape[1]
        device = input_ids.device
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[:2])
       
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransSelfAttention(nn.Module):
    def __init__(self, config: TransConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        ## 最后xshape (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 注意力加权
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TransSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TransSelfAttention(config)
        self.output = TransSelfOutput(config)

    def forward(
        self,
        hidden_states,
    ):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output


class TransIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] ## relu 

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TransOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TransAttention(config)
        self.intermediate = TransIntermediate(config)
        self.output = TransOutput(config)

    def forward(
        self,
        hidden_states
    ):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TransEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([TransLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        
        for i, layer_module in enumerate(self.layer):
            
            layer_output = layer_module(hidden_states)
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers

class InputDense2d(nn.Module):
    def __init__(self, config):
        super(InputDense2d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.in_channels, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class InputDense3d(nn.Module):
    def __init__(self, config):
        super(InputDense3d, self).__init__()
        self.dense = nn.Linear(config.patch_size[0] * config.patch_size[1] * config.patch_size[2] * config.in_channels, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = TransLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TransModel2d(nn.Module):

    def __init__(self, config):
        super(TransModel2d, self).__init__()
        self.config = config
        self.dense = InputDense2d(config)
        self.embeddings = TransEmbeddings(config)
        self.encoder = TransEncoder(config)

    def forward(
        self,
        input_ids,
        output_all_encoded_layers=True,
       
    ):  
        dense_out = self.dense(input_ids)
        embedding_output = self.embeddings(
            input_ids=dense_out
        )
        encoder_layers = self.encoder(
            embedding_output,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoder_layers[-1]
        
        if not output_all_encoded_layers:
            # 如果不用输出所有encoder层
            encoder_layers = encoder_layers[-1]
        return encoder_layers
    
class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % (sample_v**2) == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // (sample_v**2))
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v

        self.is_segmentation = is_segmentation
    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1 
        ww = w // p2 

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2)
        
        encode_x = self.bert_model(x)[-1] # 取出来最后一层
        if not self.is_segmentation:
            return encode_x

        x = self.final_dense(encode_x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)
        return encode_x, x 


class PreTrainModel(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64]):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Vit(nn.Module):
    def __init__(self, patch_size, 
                        in_channels, 
                        out_class, 
                        hidden_size=768, 
                        num_hidden_layers=12, 
                        num_attention_heads=12,
                        sample_rate=4,
                        ):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=0, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config, is_segmentation=False)
        self.cls = nn.Linear(hidden_size, out_class)

    def forward(self, x):
        encode_img = self.encoder_2d(x)
        
        encode_pool = encode_img.mean(dim=1)
        out = self.cls(encode_pool)
        return out 

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x

class SETR(nn.Module):
    def __init__(self, n_channels=3, 
                        n_classes=8, 
                        patch_size=(32, 32), 
                        hidden_size=1024, 
                        num_hidden_layers=12, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64],
                        sample_rate=4,):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=n_channels, 
                            out_channels=n_classes, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.decoder_2d = Decoder2D(in_channels=config.hidden_size, out_channels=config.out_channels, features=decode_features)

    def forward(self, x):
        _, final_x = self.encoder_2d(x)
        x = self.decoder_2d(final_x)
        return x 

class Compared_Models:
    def __init__(self):
        self.compared_models_dict = {
            "UNet": UNet,
            "PSPNet": PSPNet,
            "DeepLabV3P": DeepLabV3P,
            "SegFormer": Segformer,
            "SETR": SETR,
        }

    def get_model(self, n_channels, n_classes, model="UNet"):
        return self.compared_models_dict[model](n_channels=n_channels, n_classes=n_classes)

if __name__ == '__main__':
    compared_model = Compared_Models()
    model = "UNet"
    net = compared_model.get_model(n_channels=3, n_classes=7, model=model)
    from thop import profile
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Params: {total_params / 1e6:.2f}M")

    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 512, 512)
        flops, _ = profile(net, inputs=(input_tensor,), verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f}G")
        # net.eval()
        # data = torch.randn(2, 3, 512, 512)
        # label = net(data)
        # print(data.shape, label.shape)