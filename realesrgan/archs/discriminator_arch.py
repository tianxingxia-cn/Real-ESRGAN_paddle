from basicsr.utils.registry import ARCH_REGISTRY
# from torch import nn as nn
# from torch.nn import functional as F
# from torch.nn.utils import spectral_norm
from paddle import nn as nn
from paddle.nn import functional as F
from paddle.nn.utils import spectral_norm


@ARCH_REGISTRY.register()
# class UNetDiscriminatorSN(nn.Module):
class UNetDiscriminatorSN(nn.Layer):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # # the first convolution
        # self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # # downsample
        # self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        # self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        # self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # # upsample
        # self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        # self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        # self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # # extra convolutions
        # self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        # self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        # self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        # the first convolution
        self.conv0 = nn.Conv2D(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias_attr=True)
        # downsample
        self.conv1 = norm(nn.Conv2D(num_feat, num_feat * 2, 4, 2, 1, bias_attr=False))
        self.conv2 = norm(nn.Conv2D(num_feat * 2, num_feat * 4, 4, 2, 1, bias_attr=False))
        self.conv3 = norm(nn.Conv2D(num_feat * 4, num_feat * 8, 4, 2, 1, bias_attr=False))
        # upsample
        self.conv4 = norm(nn.Conv2D(num_feat * 8, num_feat * 4, 3, 1, 1, bias_attr=False))
        self.conv5 = norm(nn.Conv2D(num_feat * 4, num_feat * 2, 3, 1, 1, bias_attr=False))
        self.conv6 = norm(nn.Conv2D(num_feat * 2, num_feat, 3, 1, 1, bias_attr=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=False))
        self.conv8 = norm(nn.Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=False))
        self.conv9 = nn.Conv2D(num_feat, 1, 3, 1, 1, bias_attr=True)

    def forward(self, x):
        # downsample
        # x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        # x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        # x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        # x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        # x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        # x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        # out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        # out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2)
        out = self.conv9(out)

        return out
