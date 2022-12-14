import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Function


logger = logging.getLogger(__name__)

class WeldonPool2d(nn.Module):

    def __init__(self, kmax=1, kmin=None, **kwargs):
        super(WeldonPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax

        print("Using Weldon Pooling with kmax={}, kmin={}.".format(self.kmax, self.kmin))
        self._pool_func = self._define_function()

    def forward(self, input):
        return self._pool_func(input)

    def _define_function(self):
        class WeldonPool2dFunction(Function):
            @staticmethod
            def get_number_of_instances(k, n):
                if k <= 0:
                    return 0
                elif k < 1:
                    return round(k * n)
                elif k > n:
                    return int(n)
                else:
                    return int(k)

            @staticmethod
            def forward(ctx, input):
                # get batch information
                batch_size = input.size(0)
                num_channels = input.size(1)
                h = input.size(2)
                w = input.size(3)

                # get number of regions
                n = h * w

                # get the number of max and min instances
                kmax = WeldonPool2dFunction.get_number_of_instances(self.kmax, n)
                kmin = WeldonPool2dFunction.get_number_of_instances(self.kmin, n)

                # sort scores
                sorted, indices = input.new(), input.new().long()
                torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

                # compute scores for max instances
                indices_max = indices.narrow(2, 0, kmax)
                output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

                if kmin > 0:
                    # compute scores for min instances
                    indices_min = indices.narrow(2, n-kmin, kmin)
                    output.add_(sorted.narrow(2, n-kmin, kmin).sum(2).div_(kmin)).div_(2)

                # save input for backward
                ctx.save_for_backward(indices_max, indices_min, input)

                # return output with right size
                return output.view(batch_size, num_channels)

            @staticmethod
            def backward(ctx, grad_output):

                # get the input
                indices_max, indices_min, input, = ctx.saved_tensors

                # get batch information
                batch_size = input.size(0)
                num_channels = input.size(1)
                h = input.size(2)
                w = input.size(3)

                # get number of regions
                n = h * w

                # get the number of max and min instances
                kmax = WeldonPool2dFunction.get_number_of_instances(self.kmax, n)
                kmin = WeldonPool2dFunction.get_number_of_instances(self.kmin, n)

                # compute gradient for max instances
                grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)
                grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, indices_max, grad_output_max).div_(kmax)

                if kmin > 0:
                    # compute gradient for min instances
                    grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
                    grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, indices_min, grad_output_min).div_(kmin)
                    grad_input.add_(grad_input_min).div_(2)

                return grad_input.view(batch_size, num_channels, h, w)

        return WeldonPool2dFunction.apply

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax
                                                        ) + ', kmin=' + str(self.kmin) + ')'


class DownsampleStride(nn.Module):

    def __init__(self, n=2):
        super(DownsampleStride, self).__init__()
        self._n = n

    def forward(self, x):
        return x[..., ::2, ::2]


class DownsampleConv(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, stride=2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, increase_dim=False, last_relu=False, downsampling="stride"):
        super(ResidualBlock, self).__init__()

        self.increase_dim = increase_dim

        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        else:
            first_stride = 1
            planes = inplanes

        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=first_stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        if increase_dim:
            if downsampling == "stride":
                self.downsampler = DownsampleStride()
                self._need_pad = True
            else:
                self.downsampler = DownsampleConv(inplanes, planes)
                self._need_pad = False

        self.last_relu = last_relu

    @staticmethod
    def pad(x):
        return torch.cat((x, x.mul(0)), 1)

    def forward(self, x):
        y = self.conv_a(x)
        y = self.bn_a(y)
        y = F.relu(y, inplace=True)

        y = self.conv_b(y)
        y = self.bn_b(y)

        if self.increase_dim:
            x = self.downsampler(x)
            if self._need_pad:
                x = self.pad(x)

        y = x + y

        if self.last_relu:
            y = F.relu(y, inplace=True)
        
        return y


class PreActResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, increase_dim=False, last_relu=False):
        super().__init__()

        self.increase_dim = increase_dim

        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        else:
            first_stride = 1
            planes = inplanes

        self.bn_a = nn.BatchNorm2d(inplanes)
        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=first_stride, padding=1, bias=False
        )

        self.bn_b = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if increase_dim:
            self.downsample = DownsampleStride()
            self.pad = lambda x: torch.cat((x, x.mul(0)), 1)
        self.last_relu = last_relu

    def forward(self, x):
        y = self.bn_a(x)
        y = F.relu(y, inplace=True)
        y = self.conv_a(x)

        y = self.bn_b(y)
        y = F.relu(y, inplace=True)
        y = self.conv_b(y)

        if self.increase_dim:
            x = self.downsample(x)
            x = self.pad(x)

        y = x + y

        if self.last_relu:
            y = F.relu(y, inplace=True)

        return y


class Stage(nn.Module):

    def __init__(self, blocks, block_relu=False):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.block_relu = block_relu

    def forward(self, x):
        intermediary_features = []

        for b in self.blocks:
            x = b(x)
            intermediary_features.append(x)

            if self.block_relu:
                x = F.relu(x)

        return intermediary_features, x


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(
        self,
        n=5,
        nf=16,
        channels=3,
        preact=False,
        zero_residual=True,
        pooling_config={"type": "avg"},
        downsampling="stride",
        final_layer=False,
        all_attentions=False,
        last_relu=False,
        **kwargs
    ):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        if kwargs:
            raise ValueError("Unused kwargs: {}.".format(kwargs))
    
        self.all_attentions = all_attentions
        logger.info("Downsampling type {}".format(downsampling))
        self._downsampling_type = downsampling
        self.last_relu = last_relu

        Block = ResidualBlock if not preact else PreActResidualBlock

        super(CifarResNet, self).__init__()
        self.conv_1_3x3 = nn.Conv2d(channels, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(nf)

        self.layer1 = self._make_layer(Block, nf, increase_dim=False, n=n)
        self.layer2 = self._make_layer(Block, nf, increase_dim=True, n=n - 1)
        self.layer3 = self._make_layer(Block, 2 * nf, increase_dim=True, n=n - 2)
        self.layer4 = Block(
            4 * nf, increase_dim=False, last_relu=False, downsampling=self._downsampling_type
        )

        if pooling_config["type"] == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_config["type"] == "weldon":
            self.pool = WeldonPool2d(**pooling_config)
        else:
            raise ValueError("Unknown pooling type {}.".format(pooling_config["type"]))

        self.out_dim = 4 * nf
        if final_layer in (True, "conv"):
            self.fc = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False)
        elif isinstance(final_layer, dict):
            if final_layer["type"] == "one_layer":
                self.fc = nn.Sequential(
                    nn.BatchNorm1d(self.out_dim), nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, int(self.out_dim * final_layer["reduction_factor"]))
                )
                self.out_dim = int(self.out_dim * final_layer["reduction_factor"])
            elif final_layer["type"] == "two_layers":
                self.fc = nn.Sequential(
                    nn.BatchNorm1d(self.out_dim), nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, self.out_dim), nn.BatchNorm1d(self.out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, int(self.out_dim * final_layer["reduction_factor"]))
                )
                self.out_dim = int(self.out_dim * final_layer["reduction_factor"])
            else:
                raise ValueError("Unknown final layer type {}.".format(final_layer["type"]))
        else:
            self.fc = nn.Linear(self.out_dim, self.out_dim)
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if zero_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn_b.weight, 0)

    def _make_layer(self, Block, planes, increase_dim=False, n=None):
        layers = []

        if increase_dim:
            layers.append(
                Block(
                    planes,
                    increase_dim=True,
                    last_relu=False,
                    downsampling=self._downsampling_type
                )
            )
            planes = 2 * planes

        for i in range(n):
            layers.append(Block(planes, last_relu=False, downsampling=self._downsampling_type))

        return Stage(layers, block_relu=self.last_relu)

    @property
    def last_conv(self):
        return self.layer4.conv_b

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)

        feats_s1, x = self.layer1(x)
        feats_s2, x = self.layer2(x)
        feats_s3, x = self.layer3(x)
        x = self.layer4(x)

        raw_features = self.end_features(x)
        
        if self.all_attentions:
            attentions = [*feats_s1, *feats_s2, *feats_s3, x]
        else:
            attentions = [feats_s1[-1], feats_s2[-1], feats_s3[-1], x]

        # return {"raw_features": raw_features, "features": features, "attention": attentions}
        return {"features": raw_features, "fmaps": attentions}

    def end_features(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # if self.fc is not None:
        #     x = self.fc(x)

        return x


def resnet_rebuffi(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # n = 5 by default used in PODNet
    n = 5 
    model = CifarResNet(n=n, **kwargs)
    return model