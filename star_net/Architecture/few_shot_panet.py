import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PANet(nn.Module):
    """
    Parametric PANet (Prototype Alignment Network) for Few-Shot HSI Segmentation.
    - 采用 ResNet 作为特征提取器。
    - 使用“可学习的原型 (Learnable Prototypes)”替代 Support Set 提取，完美兼容标准的前向传播流程。
    - 核心机制：通过计算像素特征与原型的余弦相似度 (Cosine Similarity) 来生成分割掩膜。
    """

    def __init__(self, in_channels=110, out_channels=1, feature_dim=256):
        super(PANet, self).__init__()

        # ==========================================
        # 1. 骨干特征提取器 (Backbone)
        # 采用 ResNet50，使用空洞卷积保留较高的空间分辨率 (Output Stride = 8)
        # ==========================================
        backbone = models.resnet50(replace_stride_with_dilation=[False, True, True])

        # 针对 HSI 修改第一层卷积
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4  # 输出通道数为 2048
        )

        # 降维到指定的特征维度 (默认 256)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(2048, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # 2. 小样本灵魂：原型网络 (Prototypes)
        # ==========================================
        # 我们需要两个原型：背景原型 (0) 和 病斑原型 (1)
        # 使用 nn.Parameter 让模型在少样本微调时，记住病斑的“标准特征向量”
        self.prototypes = nn.Parameter(torch.randn(2, feature_dim, 1, 1))

        # 余弦相似度通常比较小 [-1, 1]，加一个可学习的缩放因子帮助 Softmax/Sigmoid 收敛
        self.scaler = nn.Parameter(torch.tensor(10.0))

        self.out_channels = out_channels

    def forward(self, x):
        input_size = x.shape[-2:]  # 保存原始 [H, W]

        # 1. 提取高维特征
        features = self.encoder(x)
        features = self.reduce_conv(features)  # [B, 256, H/8, W/8]

        # 2. 原型对齐 (Prototype Alignment - 余弦相似度计算)
        # 对特征和原型分别在通道维度进行 L2 归一化
        features_norm = F.normalize(features, p=2, dim=1)  # [B, 256, H/8, W/8]
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # [2, 256, 1, 1]

        # 计算余弦相似度 (Cosine Similarity)：归一化后的向量点积
        # 结果 shape: [B, 2, H/8, W/8]
        similarity = F.conv2d(features_norm, prototypes_norm) * self.scaler

        # 3. 生成二分类 Logits
        if self.out_channels == 1:
            # 取病斑相似度与背景相似度的差值，作为 Sigmoid / BCE 的 logits
            logits = similarity[:, 1:2, :, :] - similarity[:, 0:1, :, :]
        else:
            logits = similarity

        # 4. 双线性插值还原回原图大小
        out = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return out


if __name__ == "__main__":
    # 本地联调测试
    model = PANet(in_channels=110, out_channels=1, feature_dim=256)
    dummy_input = torch.randn(2, 110, 256, 256)
    output = model(dummy_input)
    print(f"✅ PANet (Few-Shot Baseline) 测试通过！")
    print(f"输入尺寸: {dummy_input.shape} -> 输出尺寸: {output.shape}")