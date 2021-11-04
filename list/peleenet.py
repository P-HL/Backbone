import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from detectron2.modeling import ShapeSpec


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, use_relu=True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out


class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)

        self.stem_2a = Conv_bn_relu(num_init_features, int(num_init_features / 2), 1, 1, 0)

        self.stem_2b = Conv_bn_relu(int(num_init_features / 2), num_init_features, 3, 2, 1)

        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem_3 = Conv_bn_relu(num_init_features * 2, num_init_features, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))

        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb1_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)

        self.cb2_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb2_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_c = Conv_bn_relu(growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)

        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp, oup, 1, 1, 0),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1, 1, 0)

    def forward(self, x):
        out = self.tb(x)
        return out


class PeleeNet(Backbone):
    def __init__(self, num_classes=1000, num_init_features=32, growthRate=32, nDenseBlocks=[3, 4, 8, 6],
                 bottleneck_width=[1, 2, 4, 4]):
        super(PeleeNet, self).__init__()

        self.stage = nn.Sequential()
        self.num_classes = num_classes
        self.num_init_features = num_init_features

        inter_channel = list()
        total_filter = list()
        dense_inp = list()

        self.half_growth_rate = int(growthRate / 2)

        # building stemblock
        self.stage.add_module('stage_0', StemBlock(3, num_init_features))

        #
        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                dense_inp.append(self.num_init_features)
            else:
                total_filter.append(total_filter[i - 1] + growthRate * nDenseBlocks[i])
                dense_inp.append(total_filter[i - 1])

            if i == len(nDenseBlocks) - 1:
                with_pooling = False
            else:
                with_pooling = True

            # building middle stageblock
            self.stage.add_module('stage_{}'.format(i + 1), self._make_dense_transition(dense_inp[i], total_filter[i],
                                                                                        inter_channel[i],
                                                                                        nDenseBlocks[i],
                                                                                        with_pooling=with_pooling))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(total_filter[len(nDenseBlocks) - 1], self.num_classes)
        )

        self._initialize_weights()

    def _make_dense_transition(self, dense_inp, total_filter, inter_channel, ndenseblocks, with_pooling=True):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel, self.half_growth_rate))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp, total_filter, with_pooling))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x=self.stage._modules['stage_0'](x)
        outputs['res2']=x
        x=self.stage._modules['stage_1'](x)
        outputs['res3']=x
        x=self.stage._modules['stage_2'](x)
        outputs['res4']=x
        x=self.stage._modules['stage_3'](x)
        x=self.stage._modules['stage_4'](x)
        outputs['res5']=x

        # x = self.stage(x)

        # global average pooling layer
        # x = F.avg_pool2d(x, kernel_size=7)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # out = F.log_softmax(x, dim=1)

        return outputs

    def output_shape(self):

        return {'res2': ShapeSpec(channels=32, stride=4),
                'res3': ShapeSpec(channels=128, stride=8),
                'res4': ShapeSpec(channels=256, stride=16),
                'res5': ShapeSpec(channels=704, stride=32)}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

@BACKBONE_REGISTRY.register()
def build_peleenet_backbone(cfg, input_shape):
    return PeleeNet()

if __name__ == '__main__':
    p = PeleeNet(num_classes=1000)
    from torchsummary import summary
    summary(p, (3, 224, 224))
    pass
    input = torch.autograd.Variable(torch.ones(1, 3, 224, 224))
    output = p(input)

    print(output.size())

    # torch.save(p.state_dict(), 'peleenet.pth.tar')