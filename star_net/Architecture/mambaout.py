import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaOutBlock(nn.Module):
    """
    CVPR 2024 MambaOut 核心块 (Gated CNN)
    通过大核深度卷积和门控机制模拟 Mamba 的序列建模能力，但保持纯 CNN 的高效。
    """

    def __init__(self, dim, expansion_ratio=8 / 3, kernel_size=7):
        super().__init__()
        # 使用 GroupNorm(1, dim) 替代 LayerNorm，避免繁琐的维度 Permute
        self.norm = nn.GroupNorm(1, dim)

        hidden_dim = int(dim * expansion_ratio)

        # 门控机制的两条分支
        self.proj1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        # 大核深度可分离卷积提取空间特征
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=hidden_dim, bias=False)

        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        # 分支1: 线性变换
        x1 = self.proj1(x)

        # 分支2: 空间特征提取 + 激活
        x2 = self.proj2(x)
        x2 = self.dwconv(x2)
        x2 = self.act(x2)

        # 门控相乘 (Gated Linear Unit)
        x = x1 * x2

        # 输出投影
        x = self.proj_out(x)
        return x + shortcut


class MambaOut_UNet(nn.Module):
    def __init__(self, in_channels=110, num_classes=1):
        super(MambaOut_UNet, self).__init__()

        # Stem (通道数: 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Encoder
        self.enc1 = MambaOutBlock(64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_conv = nn.Conv2d(64, 128, kernel_size=1)  # 升维
        self.enc2 = MambaOutBlock(128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_conv = nn.Conv2d(128, 256, kernel_size=1)
        self.enc3 = MambaOutBlock(256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_conv = nn.Conv2d(256, 512, kernel_size=1)
        self.enc4 = MambaOutBlock(512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_conv = nn.Conv2d(512, 256, kernel_size=1)  # 拼接后降维
        self.dec3 = MambaOutBlock(256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.dec2 = MambaOutBlock(128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.dec1 = MambaOutBlock(64)

        # Output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)

        e1 = self.enc1(x0)

        e2 = self.pool1(e1)
        e2 = self.enc2(self.enc2_conv(e2))

        e3 = self.pool2(e2)
        e3 = self.enc3(self.enc3_conv(e3))

        e4 = self.pool3(e3)
        e4 = self.enc4(self.enc4_conv(e4))

        # Skip Connections
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(self.dec3_conv(d3))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(self.dec2_conv(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(self.dec1_conv(d1))

        return self.final_conv(d1)