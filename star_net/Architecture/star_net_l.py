import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 改进版：自适应高频感知 DCT 多谱通道注意力
# ==========================================
class DCTMultiSpectralSSA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(DCTMultiSpectralSSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 【修改 1：扩展频率网格】
        # 使用 4x4 的频率坐标 (u,v)，涵盖从 (0,0) 直流低频到 (3,3) 的中高频
        self.dct_freqs = [(u, v) for u in range(4) for v in range(4)]

        # 【修改 2：引入自适应频率权重】
        # 定义 16 个可学习参数，初始值为 1.0。
        # 训练完成后，取出这个权重即可作为论文可视化的核心证据！
        self.freq_weights = nn.Parameter(torch.ones(len(self.dct_freqs)))

        self._basis_cache = {}

    def _get_dct_basis(self, H, W, u, v, device):
        key = (H, W, u, v)
        if key not in self._basis_cache:
            h_idx = torch.arange(H, device=device).float()
            w_idx = torch.arange(W, device=device).float()
            h_basis = torch.cos(math.pi * u * (h_idx + 0.5) / H)
            w_basis = torch.cos(math.pi * v * (w_idx + 0.5) / W)
            self._basis_cache[key] = (h_basis.unsqueeze(1) * w_basis.unsqueeze(0)).detach()
        return self._basis_cache[key].to(device)

    def _dct_pool(self, x, u, v):
        B, C, H, W = x.shape
        basis = self._get_dct_basis(H, W, u, v, x.device)
        return (x * basis.unsqueeze(0).unsqueeze(0)).sum(dim=[2, 3], keepdim=True) / (H * W)

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        # 【修改 3：动态应用自适应权重】
        dct_out = 0
        for idx, (u, v) in enumerate(self.dct_freqs):
            freq_proj = self._dct_pool(x, u, v)
            # 乘以网络自己学到的该频率分量的重要性权重
            weighted_proj = freq_proj * self.freq_weights[idx]
            dct_out = dct_out + self.mlp(weighted_proj)

        spectral_weight = self.sigmoid(avg_out + max_out + dct_out)
        spectral_out = spectral_weight * x

        spatial_avg = torch.mean(spectral_out, dim=1, keepdim=True)
        spatial_max, _ = torch.max(spectral_out, dim=1, keepdim=True)
        spatial_pool = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_pool))

        return spatial_weight * spectral_out


# ==========================================
# 2. Ghost 模块
# ==========================================
class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, ratio=2):
        super(GhostModule, self).__init__()
        self.primary_out = out_ch // ratio
        self.ghost_out = out_ch - self.primary_out
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, self.primary_out, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.primary_out),
        )
        if self.ghost_out > 0:
            self.ghost_conv = nn.Sequential(
                nn.Conv2d(self.primary_out, self.ghost_out, 3, 1, 1,
                          groups=self.primary_out, bias=False),
                nn.BatchNorm2d(self.ghost_out),
            )
        else:
            self.ghost_conv = None

    def forward(self, x):
        primary = self.primary_conv(x)
        if self.ghost_conv is not None:
            ghost = self.ghost_conv(primary)
            return torch.cat([primary, ghost], dim=1)
        return primary


# ==========================================
# 3. Ghost 残差块 — Encoder 用
# ==========================================
class GhostResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, ratio=2):
        super(GhostResBlock, self).__init__()
        self.ghost1 = GhostModule(in_channels, out_channels, 3, stride, ratio)
        self.relu1 = nn.ReLU(inplace=True)
        self.ghost2 = GhostModule(out_channels, out_channels, 3, 1, ratio)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.ghost1(x))
        out = self.ghost2(out)
        return F.relu(out + identity, inplace=True)


# ==========================================
# 4. 自适应残差块 — Decoder 用
# ==========================================
class AdaptiveGhostResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, ratio=2):
        super(AdaptiveGhostResBlock, self).__init__()
        self.ghost1 = GhostModule(in_channels, out_channels, 3, stride, ratio)
        self.relu1 = nn.ReLU(inplace=True)
        self.ghost2 = GhostModule(out_channels, out_channels, 3, 1, ratio)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.res_gate = nn.Parameter(torch.tensor(0.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.ghost1(x))
        out = self.ghost2(out)
        alpha = self.sigmoid(self.res_gate)
        return F.relu(out + alpha * identity, inplace=True)


# ==========================================
# 5. UniversalEncoder
# ==========================================
class UniversalEncoder(nn.Module):
    def __init__(self, in_channels=110):
        super(UniversalEncoder, self).__init__()
        self.down1 = GhostResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = GhostResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = GhostResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = GhostResBlock(256, 512)
        self.bottleneck_att = DCTMultiSpectralSSA(512)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        x4 = self.bottleneck_att(x4)
        return x1, x2, x3, x4


class DiseaseSpecificDecoderGrape(nn.Module):
    def __init__(self, out_channels=1):
        super(DiseaseSpecificDecoderGrape, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = DCTMultiSpectralSSA(256)
        self.up_conv1 = GhostResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = DCTMultiSpectralSSA(128)
        self.up_conv2 = GhostResBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = DCTMultiSpectralSSA(64)
        self.up_conv3 = GhostResBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        up1 = self.up1(x4)
        x = self.up_conv1(torch.cat([up1, self.att1(x3)], dim=1))
        up2 = self.up2(x)
        x = self.up_conv2(torch.cat([up2, self.att2(x2)], dim=1))
        up3 = self.up3(x)
        x = self.up_conv3(torch.cat([up3, self.att3(x1)], dim=1))
        return self.out_conv(x)


class DiseaseSpecificDecoderAdaptive(nn.Module):
    def __init__(self, out_channels=1):
        super(DiseaseSpecificDecoderAdaptive, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = DCTMultiSpectralSSA(256)
        self.up_conv1 = AdaptiveGhostResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = DCTMultiSpectralSSA(128)
        self.up_conv2 = AdaptiveGhostResBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = DCTMultiSpectralSSA(64)
        self.up_conv3 = AdaptiveGhostResBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        up1 = self.up1(x4)
        x = self.up_conv1(torch.cat([up1, self.att1(x3)], dim=1))
        up2 = self.up2(x)
        x = self.up_conv2(torch.cat([up2, self.att2(x2)], dim=1))
        up3 = self.up3(x)
        x = self.up_conv3(torch.cat([up3, self.att3(x1)], dim=1))
        return self.out_conv(x)


# ==========================================
# 6. 主模型：STARNetL
# ==========================================
class STARNetL(nn.Module):
    def __init__(self, in_channels=110, tasks=['grape', 'corn', 'tomato']):
        super(STARNetL, self).__init__()
        self.tasks = tasks
        self.encoder = UniversalEncoder(in_channels)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, len(tasks))
        )
        self.decoders = nn.ModuleDict({
            'grape': DiseaseSpecificDecoderGrape(),
            'corn': DiseaseSpecificDecoderGrape(),
            'tomato': DiseaseSpecificDecoderAdaptive(),
        })

    def forward(self, x, is_training=False, target_task_name=None):
        x1, x2, x3, x4 = self.encoder(x)
        task_logits = self.router(x4)
        if is_training:
            mask_logits = self.decoders[target_task_name](x1, x2, x3, x4)
            return mask_logits, task_logits
        else:
            pred_task_id = torch.argmax(task_logits, dim=1)[0].item()
            pred_task_name = self.tasks[pred_task_id]
            mask_logits = self.decoders[pred_task_name](x1, x2, x3, x4)
            return mask_logits, pred_task_name