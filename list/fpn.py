# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from typing import List, Optional
#import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
#from resnet import build_resnet_backbone
#from mobilenetV3large import build_mobilenetV3large_backbone
#from CSPdarknet53 import build_CSPdarknet53_backbone
from .mobilenetV3small import build_mobilenetV3small_backbone

__all__ = ["build_resnet_fpn_backbone", "build_retinanet_resnet_fpn_backbone", "build_mobilenetV3large_fpn_backbone","build_mobilenetV3small_fpn_backbone","FPN","build_CSPdarknet53_FPN_backbone"]

##################################画图####################################
def Have_a_Look(image,str):
    print(str)
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(image.detach().cpu().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    im_mean = np.mean(im,axis=(2))

    # 查看这一层不同通道的图像，在这里有256层
    plt.figure()
    # for i in range(16):
    #     ax = plt.subplot(4, 4, i+1)
    #     plt.suptitle(str)
    #     plt.imshow(im[:, :, i], cmap='gray')
    plt.show()
#########################################################################

#######################################################################
#定义瓶颈结构
class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels=16,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        #self.se = SELayer1(out_channels)

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        '''
        Arge:
            # Zero-initialize the last normalization in each residual branch,
            # so that at the beginning, the residual branch starts with zeros,
            # and each residual block behaves like an identity.
            # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "For BN layers, the learnable scaling coefficient γ is initialized
            # to be 1, except for each residual block's last BN
            # where γ is initialized to be 0."

            # nn.init.constant_(self.conv3.norm.weight, 0)
            # TODO this somehow hurts performance when training GN models from scratch.
            # Add it as an option when we need to use this code to train a backbone.
        '''

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        # if y is not None:
        #     inrelu = self.se(y) #注意力快捷方式
        # else:
        #     inrelu = self.shortcut(x)
        
        # # if self.shortcut is not None:
        # #     shortcut = self.shortcut(x)
        # #     #shortcut = self.se(shortcut) #捷径添加注意力
        # # else:
        # #     shortcut = x
        # #     #shortcut = self.se(x) #捷径添加注意力

        # # out += shortcut
        # # out = F.relu_(out)  #将最后两步留到外面
        return out
#########################################################################

#######################################################################
#定义SE注意力模块 参考：https://blog.csdn.net/weixin_42907473/article/details/106525668
class SELayer(nn.Module):
    def __init__(self, in_channels,channel,bias,norm, reduction=16):
        super(SELayer, self).__init__()
        self.conv1 = Conv2d(in_channels, channel, kernel_size=1, bias=bias, norm=norm)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
######################################################################
#定义无1X1卷积SE注意力：
'''
class SELayer1(nn.Module):
    def __init__(self, channel=256, reduction=16):
        super(SELayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
'''

#定义注意力瓶颈结构
'''
class SEBottleneck(nn.Module):
        expansion = 4

        #def __init__(self, in_channels, out_channels, bias, norm, downsample=None, reduction=16):
        def __init__(self, in_channels, out_channels, bias, norm,  reduction=16):
            super(SEBottleneck, self).__init__()
            self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, norm=norm)
            self.se = SELayer(out_channels, reduction)

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.se(out)

            # if self.downsample is not None:
            #     residual = self.downsample(x)
            
            out += residual
            out = self.relu(out)

            return out
'''
#######################################################################

#定义CA注意力模块
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w
        print(self.h)
        print(self.w)

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0,1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

'''
#定义deeplab
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)
'''

#定义清晰的deeplab
class ASPP(nn.Module):
    def __init__(self, num_classes,in_channels):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, dilation=1) #修改一下？1,2,5,第一次实验是6,12,18
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=2, dilation=2)#保持原图大小不变 padding=dilation
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        #self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        #self.conv_1x1_3 = nn.Conv2d(768, 256, kernel_size=1) # (768 = 3*256)
        self.conv_1x1_3 = nn.Conv2d(1024, 256, kernel_size=1) # (1024 = 4*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    '''
        def forward(self, feature_map):
            # (feature_map has shape (batch_size, 2048, h/8, w/8))

            feature_map_h = feature_map.size()[2] # (h/8)
            feature_map_w = feature_map.size()[3] # (w/8)
            #print("+++++++++++++++++++++++++++++++++膨胀卷积输入的h",feature_map_h)  #可以使用
            #print(feature_map_w)
            out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 E
            out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 D
            out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 C
            out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 B

            #out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))对应图中 ImagePooling

            #out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1)) 
            #out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/8, w/8))对应图中 A

            #out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/8, w/8)) cat对应图中 F
            out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3], 1) #将张量按照一维拼接
            out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/8, w/8)) bn_conv_1x1_3对应图中 H  out 对应图中I
            #out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/8, w/8))out 对应图中Upsample by 4

            return out
    '''
    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 2048, h/8, w/8))

        ##feature_map_w = feature_map.size()[3] # (w/8)
        #print("+++++++++++++++++++++++++++++++++膨胀卷积输入的h",feature_map_h)  #可以使用
        #print(feature_map_w)
        #out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 E
        #out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 D
        #out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 C
        #out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/8, w/8)) 对应图中 B
        #--------------------#
        #   连续空洞卷积
        #--------------------#
        out = self.conv_3x3_1(feature_map)
        out = self.conv_3x3_2(out)
        out = self.conv_3x3_3(out)
        #out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))对应图中 ImagePooling

        #out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1)) 
        #out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/8, w/8))对应图中 A

        #out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/8, w/8)) cat对应图中 F
        #out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3], 1) #将张量按照一维拼接
        #out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/8, w/8)) bn_conv_1x1_3对应图中 H  out 对应图中I
        #out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/8, w/8))out 对应图中Upsample by 4

        return out



class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        

        #h = [input_shapes[f].height for f in in_features]
        #w = [input_shapes[f].width for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        #in_My_channels = 0
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)
            stage = int(math.log2(strides[idx]))
        ##----------------------------------------------------------------------------------------------##
            ##横向链接1X1，256卷积
            #lateral_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm)
            #lateral_conv = SEBottleneck(in_channels, out_channels, bias=use_bias, norm=lateral_norm)

            #加入含1X1卷积的注意力模块
            lateral_conv = SELayer(in_channels, out_channels, bias=use_bias, norm=lateral_norm)
            
            
            #加入CA注意力模块
            #lateral_conv = CA_Block(channel=in_channels, h=h[idx], w=w[idx])

            #组合拳 
            #weight_init.c2_xavier_fill(lateral_conv)
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            lateral_convs.append(lateral_conv)  
            #$lateral_convs.append(aspp_conv)          
        ##---------------------------------------------------------------------------------------------##

            #将一个3*3卷积换成1*1->3*3->1*1，加深网络
            #output_conv = Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=use_bias,norm=output_norm,)

            #替换卷积
            output_conv = BottleneckBlock(out_channels,out_channels)

            #weight_init.c2_xavier_fill(output_conv)    #这一步老是报错
            self.add_module("fpn_output{}".format(stage), output_conv)
            output_convs.append(output_conv)
            



            
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.

        self.lateral_convs = lateral_convs[::-1] #翻转上下层的卷积
        self.output_convs = output_convs[::-1] #翻转上下层的卷积
        #self.lateral_convs = lateral_convs[::1] #不翻转卷积
        #self.output_convs = output_convs[::1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        # Scripting does not support this: https://github.com/pytorch/pytorch/issues/47334
        # have to do it in __init__ instead.

        self.rev_in_features = tuple(in_features[::-1]) #翻转结构

        #self.rev_in_features = self.in_features #不翻转
        #self.se = SELayer1(channel=256,reduction=16)

        #self.atrous_rates = (6, 12, 18)  
        #self.aspp_conv = ASPP(atrous=self.atrous_rates)

        #self.deeplab = ASPP(576, [6, 12, 18])

        self.deeplab = ASPP(256,576) #虽然真正的num_lass是2，但是为了能够与后面的fpn结合，这里写为256
        self.deeplab_all=[]
        self.deeplab1 = ASPP(256,48)
        self.deeplab2 = ASPP(256,24)
        self.deeplab3 = ASPP(256,256)
        #self.deeplab_all.append(self.deeplab)
        self.deeplab_all.append(self.deeplab1)
        self.deeplab_all.append(self.deeplab2)
        
        

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]]) #in_features[-1] 最顶层的图像
        
        #顶层图像向上deeplab
        #top_deeplab = self.aspp_conv(in_My_features = bottom_up_features[self.in_features[-1]])

        #首层加入deeplab的Encoding,第一种
        #aspp = ASPP(in_My_features = bottom_up_features[self.in_features[-1]],atrous=self.atrous_rates,in_My_channels=576)
        #aspp = ASPP(in_My_features = bottom_up_features[self.in_features[-1]],atrous=self.atrous_rates,in_My_channels=576).cuda()
 
        #top_deeplab = ASPP(in_My_features = bottom_up_features[self.in_features[-1]],atrous=self.atrous_rates,in_My_channels=576).forward()
        #top_deeplab = aspp.forward()

        #top_deeplab = self.deeplab(bottom_up_features[self.in_features[-1]])

        #shout_features = self.deeplab3(prev_features) #顶层不要deeplab
        results.append(self.output_convs[0](prev_features)+prev_features)

        #results.append(self.output_convs[0](prev_features))
        #Have_a_Look(prev_features,'C5')

        #######################################################################################################################
        #从下往上
            # for features, lateral_conv, output_conv in zip(
            #     self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:]
            # ):
            #     features = bottom_up_features[features]
            #     #Have_a_Look(features,'C4 or C3')
            #     top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest") #下采样函数
            #     '''
            #     interpolate_Args:
            #         上采样函数解释参考：https://www.jianshu.com/p/dc0d44911c6c
            #         def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
            #         根据给定 size 或 scale_factor，上采样或下采样输入数据input.
            #         参数:
            #         - input (Tensor): input tensor
            #         - size 或 scale_factor (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):输出的 spatial 尺寸.
            #         - scale_factor (float or Tuple[float]): spatial 尺寸的缩放因子.
            #         - mode (string): 上采样算法:nearest, linear, bilinear, trilinear, area. 默认为 nearest.
            #         - align_corners (bool, optional): 如果 align_corners=True，则对齐 input 和 output 的角点像素(corner pixels)，保持在角点像素的值. 只会对 mode=linear, bilinear 和 trilinear 有作用. 默认是 False.
            #     '''
            #     # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
            #     lateral_features = lateral_conv.forward(features) #横向处理之后生成的图像
            #     #Have_a_Look(lateral_features,'横向处理之后')
            #     prev_features = lateral_features + top_down_features #横向图像 与 上采样图像 相加
            #     #Have_a_Look(prev_features,'与上层相加后')
            #     if self._fuse_type == "avg":
            #         prev_features /= 2

            #     results.insert(0, output_conv.forward(prev_features)) #output输出了
        #######################################################################################################################

        
        for features, lateral_conv, output_conv in zip(
            self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            features = bottom_up_features[features]
            #lateral_features_last = lateral_conv.forward(features_last)
            #print("++++++++++++++++++++++++===================输入图片的h====================",prev_features.size()[2])
            #prev_features = self.deeplab3.forward(prev_features) #upsample之前加入ASPP
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest") #上采样函数

            # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
            lateral_features = lateral_conv.forward(features) #横向处理之后生成的图像

            #lateral_up_features = F.interpolate(lateral_features, scale_factor=0.5, mode="nearest") #下采样函数

            prev_features = lateral_features + top_down_features #横向图像 与 上采样图像 相加

            last_features = output_conv.forward(prev_features) #横向bottleNeck主路径处理
            if self._fuse_type == "avg":
                prev_features /= 2
            
            #top_down_SAPP_features = self.deeplab3.forward(top_down_features) #双向相加的bottleNeck快捷方式
            #last_features += top_down_SAPP_features #bottleNeck相加处理
            #prev_features = F.relu_(last_features) 上一个改进输出错了...没有经过relu
            #last_features = F.relu_(last_features)

            #shoutcut_features = self.deeplab3.forward(prev_features)
            shoutcut_features = self.deeplab3.forward(top_down_features) #MyFPN_Dconv
            last_features += shoutcut_features
            results.insert(0,last_features) #output输出了
    #########################################################################################################################

    #############################################################################################################################
    #原来的：
            # Reverse feature maps into top-down order (from low to high resolution)
            # 从低到高分辨率倒置特征映射到自上而下的顺序
        # for features, lateral_conv, output_conv ,deeplab in zip(
        #     self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:] ,self.deeplab_all[0:]
        # ):
        #     features = bottom_up_features[features]
        #     #Have_a_Look(features,'C4 or C3')
        #     top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest") #上采样函数
        #     '''
        #     interpolate_Args:
        #         上采样函数解释参考：https://www.jianshu.com/p/dc0d44911c6c
        #         def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
        #         根据给定 size 或 scale_factor，上采样或下采样输入数据input.
        #         参数:
        #         - input (Tensor): input tensor
        #         - size 或 scale_factor (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):输出的 spatial 尺寸.
        #         - scale_factor (float or Tuple[float]): spatial 尺寸的缩放因子.
        #         - mode (string): 上采样算法:nearest, linear, bilinear, trilinear, area. 默认为 nearest.
        #         - align_corners (bool, optional): 如果 align_corners=True，则对齐 input 和 output 的角点像素(corner pixels)，保持在角点像素的值. 只会对 mode=linear, bilinear 和 trilinear 有作用. 默认是 False.
        #     '''
        #     # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
        #     lateral_features = lateral_conv.forward(features) #横向处理之后生成的图像

        #     #加入Encoder_small快捷方式
        #     Encoder_small_features = deeplab.forward(features)

        #     #Have_a_Look(lateral_features,'横向处理之后')
        #     #prev_features = lateral_features + top_down_features #横向图像 与 上采样图像 相加
        #     prev_features = lateral_features + top_down_features + Encoder_small_features #横向图像 + 上采样图像 + 快捷方式图像
        #     #Have_a_Look(prev_features,'与上层相加后')
        #     if self._fuse_type == "avg":
        #         prev_features /= 2

        #     results.insert(0, output_conv.forward(prev_features)) #output输出了

        

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)
        return dict(list(zip(self._out_features, results)))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

##------------自己添加只有p6层----------------##
class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1 #表示增加的额外 FPN 级别的数量
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]
##-----------------------------------------##

class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2 #表示增加的额外 FPN 级别的数量
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


##----------注册FPN的backbone-----------##
'''
@BACKBONE_REGISTRY.register()
def build_mobilenetV3large_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mobilenetV3large_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        #-------直接吧top_block改成None------##
        #top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        top_block=None,
        ##------------------------------##
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
'''
@BACKBONE_REGISTRY.register()
def build_mobilenetV3small_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mobilenetV3small_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    #in_channels_p6p7 = bottom_up.output_shape()["res5"].channels

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        #-------直接吧top_block改成None------##
        #top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        top_block=None,
        ##------------------------------##
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

'''
##----------注册FPN的backbone-----------##
@BACKBONE_REGISTRY.register()
def build_CSPdarknet53_FPN_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_CSPdarknet53_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        #-------直接吧top_block改成None------##
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        #top_block=None,
        ##------------------------------##
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """

    bottom_up = build_resnet_backbone(cfg, input_shape)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

##-----------再带------------##
@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
'''