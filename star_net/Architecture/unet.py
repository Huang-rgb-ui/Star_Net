import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    U-Net 核心组件：连续两次的 卷积 -> 批归一化 -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    原汁原味的标准 U-Net 基线模型 (Modified for HSI)
    - 保留经典的 4层下采样 + 1层瓶颈 + 4层上采样 结构。
    - 保留基于通道拼接 (Concatenation) 的跳跃连接 (Skip Connections)。
    """

    def __init__(self, in_channels=110, out_channels=1, base_features=64):
        super(UNet, self).__init__()

        # ==========================================
        # Encoder (下采样路径)
        # ==========================================
        self.inc = DoubleConv(in_channels, base_features)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features, base_features * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features * 2, base_features * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features * 4, base_features * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_features * 8, base_features * 16)
        )

        # ==========================================
        # Decoder (上采样路径)
        # 原汁原味的 U-Net 使用转置卷积 (ConvTranspose2d) 进行学习型上采样
        # ==========================================
        self.up1 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 16, base_features * 8)

        self.up2 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 8, base_features * 4)

        self.up3 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 4, base_features * 2)

        self.up4 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_features * 2, base_features)

        # 最终输出层
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. 编码提取特征 (并保留跳跃连接的特征图)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck 瓶颈层

        # 2. 解码与特征融合
        # 第一层上采样：把 x5 放大，然后和 x4 拼接
        x = self.up1(x5)
        x = self._pad_and_cat(x, x4)
        x = self.conv_up1(x)

        # 第二层上采样
        x = self.up2(x)
        x = self._pad_and_cat(x, x3)
        x = self.conv_up2(x)

        # 第三层上采样
        x = self.up3(x)
        x = self._pad_and_cat(x, x2)
        x = self.conv_up3(x)

        # 第四层上采样
        x = self.up4(x)
        x = self._pad_and_cat(x, x1)
        x = self.conv_up4(x)

        # 3. 输出病斑分类掩膜
        logits = self.outc(x)
        return logits

    def _pad_and_cat(self, x1, x2):
        """
        处理高光谱图像剪裁或下采样过程中可能出现的奇数边缘问题。
        确保特征图拼接时尺寸完全一致。
        """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 如果因为 MaxPool 导致尺寸差了 1 个像素，自动补齐
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 将深层上采样特征与浅层跳跃特征在通道维度 (dim=1) 进行拼接
        return torch.cat([x2, x1], dim=1)


if __name__ == "__main__":
    # 简单的本地联调测试，确保网络没有写错
    model = UNet(in_channels=110, out_channels=1)
    dummy_input = torch.randn(1, 110, 256, 256)
    output = model(dummy_input)
    print(f"✅ U-Net 测试通过！输入尺寸: {dummy_input.shape} -> 输出尺寸: {output.shape}")