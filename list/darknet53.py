import torch
from torch import nn
import torch.nn.functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from detectron2.modeling import ShapeSpec


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding):
        super(ConvolutionalLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.reseblock = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, kernal_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels // 2, in_channels, kernal_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.reseblock(x)


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernal_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvolutionalSetLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channel, out_channel, kernal_size=1, stride=1, padding=0),
            ConvolutionalLayer(out_channel, in_channel, kernal_size=3, stride=1, padding=1),
            ConvolutionalLayer(in_channel, out_channel, kernal_size=1, stride=1, padding=0),
            ConvolutionalLayer(out_channel, in_channel, kernal_size=3, stride=1, padding=1),
            ConvolutionalLayer(in_channel, out_channel, kernal_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x)


class DarkNet53(Backbone):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.feature_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),
            ResidualLayer(64),
            DownSampleLayer(64, 128),
            ResidualLayer(128),
            ResidualLayer(128),
            DownSampleLayer(128, 256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )
        self.feature_26 = nn.Sequential(
            DownSampleLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )
        self.feature_13 = nn.Sequential(
            DownSampleLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.convolset_13 = nn.Sequential(
            ConvolutionalSetLayer(1024, 512)
        )
        self.convolset_26 = nn.Sequential(
            ConvolutionalSetLayer(768, 256)
        )
        self.convolset_52 = nn.Sequential(
            ConvolutionalSetLayer(384, 128)
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 15, 1, 1, 0)
        )
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 15, 1, 1, 0)
        )
        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 15, 1, 1, 0)
        )
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpSampleLayer()
        )
        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpSampleLayer()
        )

    def forward(self, x):
        outputs = {}
        outlist = ['5', '14']
        outnames = ['res2', 'res3']
        ptr = 0
        for i in range(len(self.feature_52)):
            x = self.feature_52._modules[str(i)](x)
            if str(i) in outlist:
                outputs[outnames[ptr]] = x
                ptr += 1

        h_26 = self.feature_26(x)
        outputs['res4'] = h_26

        h_13 = self.feature_13(h_26)
        outputs['res5'] = h_13

        return outputs

    def output_shape(self):

        return {'res2': ShapeSpec(channels=128, stride=4),
                'res3': ShapeSpec(channels=256, stride=8),
                'res4': ShapeSpec(channels=512, stride=16),
                'res5': ShapeSpec(channels=1024, stride=32)}


@BACKBONE_REGISTRY.register()
def build_darknet53_backbone(cfg, input_shape):
    return DarkNet53()


if __name__ == '__main__':
    net = DarkNet53()
    from torchsummary import summary

    summary(net, (3, 224, 224))
    pass
