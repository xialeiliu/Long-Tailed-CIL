import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet32']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(ResNetBasicblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.last_relu:
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class CifarResNet(nn.Module):
    """CifarResNet that resembles closer to other FACIL method architectures. 
    The resnet rebuffi originally used by PODNet contains an extra layer of Basic block making comparisons unfair!
    """
    def __init__(self, block, depth, num_classes=10):
        self.inplanes = 16
        super(CifarResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layer_blocks, last_relu=False)
        self.layer2 = self._make_layer(block, 32, layer_blocks, stride=2, last_relu=False)
        self.layer3 = self._make_layer(block, 64, layer_blocks, stride=2, last_relu=False)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64*block.expansion
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, last_relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, last_relu=last_relu))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, last_relu=last_relu))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # x = [128, 16, 32, 32]
        x_1 = self.layer1(x) # x_1 = [128, 16, 32, 32]
        x_2 = self.layer2(x_1) # x_2 = [128, 32, 16, 16]
        x_3 = self.layer3(x_2) # x_3 = [128, 64, 8, 8]
        raw_features = self.end_features(x_3) # [128, 64] as it flattens the spatial coordinates, therefore requiring POD-flat losss

        return {
            'fmaps': [x_1, x_2, x_3],
            'features': raw_features
        }

    def end_features(self, x):
        x = self.avgpool(x) # transforms [128, 64, 8, 8] into [128, 64, 1, 1]
        x = x.view(x.size(0), -1) # transforms [128, 64, 1, 1] into [128, 64]
        return x

def resnet32_podnet(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = CifarResNet(ResNetBasicblock, 32, **kwargs)
    return model
