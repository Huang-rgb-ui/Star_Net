import torch
import torch.nn as nn
import torch.nn.functional as F


class PConvBlock(nn.Module):
    """Partial Convolution Block (Chen et al., CVPR 2023)
       Only convolve cp channels, leave rest as identity (T-shaped)"""
    def __init__(self, dim, expand_ratio=2):
        super(PConvBlock, self).__init__()
        cp = dim // 4  # partial channels
        hidden = dim * expand_ratio
        self.pconv = nn.Conv2d(cp, cp, 3, padding=1, bias=False)
        self.pwconv1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        # PConv: only first cp channels get convolved
        cp = x.shape[1] // 4
        x_cp = self.pconv(x[:, :cp])
        x_out = torch.cat([x_cp, x[:, cp:]], dim=1)
        # PWConv
        x_out = self.pwconv1(x_out)
        x_out = self.act(x_out)
        x_out = self.pwconv2(x_out)
        return x + x_out


class FasterNetUNet(nn.Module):
    """UNet with FasterNet PConv blocks, ~4M params"""
    def __init__(self, in_channels=110, base_ch=64):
        super(FasterNetUNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )
        self.enc1 = PConvBlock(base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.trans1 = nn.Conv2d(base_ch, base_ch * 2, 1, bias=False)
        self.enc2 = PConvBlock(base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.trans2 = nn.Conv2d(base_ch * 2, base_ch * 4, 1, bias=False)
        self.enc3 = PConvBlock(base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.trans3 = nn.Conv2d(base_ch * 4, base_ch * 8, 1, bias=False)
        self.bottleneck = PConvBlock(base_ch * 8)

        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec1 = PConvBlock(base_ch * 8)
        self.proj1 = nn.Conv2d(base_ch * 8, base_ch * 4, 1, bias=False)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = PConvBlock(base_ch * 4)
        self.proj2 = nn.Conv2d(base_ch * 4, base_ch * 2, 1, bias=False)
        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec3 = PConvBlock(base_ch * 2)
        self.proj3 = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.trans1(self.pool1(e1)))
        e3 = self.enc3(self.trans2(self.pool2(e2)))
        b = self.bottleneck(self.trans3(self.pool3(e3)))
        d1 = self.proj1(self.dec1(torch.cat([self.up1(b), e3], dim=1)))
        d2 = self.proj2(self.dec2(torch.cat([self.up2(d1), e2], dim=1)))
        d3 = self.proj3(self.dec3(torch.cat([self.up3(d2), e1], dim=1)))
        return self.out_conv(d3)
