import torch
import torch.nn as nn
import torch.nn.functional as F


class PKIBlock(nn.Module):
    """
    Poly-Kernel Inception Block 简化版
    使用不同大小的卷积核提取多尺度特征
    """

    def __init__(self, in_channels, out_channels):
        super(PKIBlock, self).__init__()
        mid_channels = out_channels // 4

        # 1x1 Conv 降维
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(mid_channels)

        # 3x3 Conv
        self.conv3x3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(mid_channels)

        # 5x5 Conv
        self.conv5x5 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False)
        self.bn5x5 = nn.BatchNorm2d(mid_channels)

        # 7x7 Conv (可用膨胀卷积替代以减少参数)
        self.conv7x7 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn7x7 = nn.BatchNorm2d(mid_channels)

        self.relu = nn.ReLU(inplace=True)

        # 特征融合
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.relu(self.bn1x1(self.conv1x1(x)))
        out3 = self.relu(self.bn3x3(self.conv3x3(x)))
        out5 = self.relu(self.bn5x5(self.conv5x5(x)))
        out7 = self.relu(self.bn7x7(self.conv7x7(x)))

        out = torch.cat([out1, out3, out5, out7], dim=1)
        out = self.fuse_bn(self.fuse_conv(out))

        out += self.shortcut(x)
        return self.relu(out)


class PKINet(nn.Module):
    def __init__(self, in_channels=110, num_classes=1):
        super(PKINet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder (Downsampling)
        self.enc1 = PKIBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = PKIBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = PKIBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = PKIBlock(256, 512)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = PKIBlock(512, 256)  # 256 (up) + 256 (skip)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = PKIBlock(256, 128)  # 128 (up) + 128 (skip)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = PKIBlock(128, 64)  # 64 (up) + 64 (skip)

        # Output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)

        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.final_conv(d1)
        return out