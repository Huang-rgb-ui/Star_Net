import torch
import torch.nn as nn
import torch.nn.functional as F


class RepVGGBlock(nn.Module):
    """RepVGG Block (Ding et al., CVPR 2021)
       Train: 3x3 + 1x1 + Identity branches
       Here: single 3x3 path (reparam-equivalent), with residual"""
    def __init__(self, dim):
        super(RepVGGBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + x)


class RepVGGUNet(nn.Module):
    """UNet with RepVGG blocks, ~7M params"""
    def __init__(self, in_channels=110, base_ch=64):
        super(RepVGGUNet, self).__init__()
        C = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.enc1 = RepVGGBlock(C)
        self.pool1 = nn.MaxPool2d(2)
        self.t1 = nn.Conv2d(C, C * 2, 1, bias=False)
        self.enc2 = RepVGGBlock(C * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.t2 = nn.Conv2d(C * 2, C * 4, 1, bias=False)
        self.enc3 = RepVGGBlock(C * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.t3 = nn.Conv2d(C * 4, C * 8, 1, bias=False)
        self.bottleneck = RepVGGBlock(C * 8)

        self.up1 = nn.ConvTranspose2d(C * 8, C * 4, 2, 2)
        self.dec1 = RepVGGBlock(C * 8)
        self.proj1 = nn.Conv2d(C * 8, C * 4, 1, bias=False)
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, 2)
        self.dec2 = RepVGGBlock(C * 4)
        self.proj2 = nn.Conv2d(C * 4, C * 2, 1, bias=False)
        self.up3 = nn.ConvTranspose2d(C * 2, C, 2, 2)
        self.dec3 = RepVGGBlock(C * 2)
        self.proj3 = nn.Conv2d(C * 2, C, 1, bias=False)
        self.out_conv = nn.Conv2d(C, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.t1(self.pool1(e1)))
        e3 = self.enc3(self.t2(self.pool2(e2)))
        b = self.bottleneck(self.t3(self.pool3(e3)))
        d1 = self.proj1(self.dec1(torch.cat([self.up1(b), e3], dim=1)))
        d2 = self.proj2(self.dec2(torch.cat([self.up2(d1), e2], dim=1)))
        d3 = self.proj3(self.dec3(torch.cat([self.up3(d2), e1], dim=1)))
        return self.out_conv(d3)
