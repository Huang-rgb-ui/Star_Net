import torch
import torch.nn as nn
import torch.nn.functional as F

# 重型模型
class SpectralSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SpectralSpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        spectral_weight = self.sigmoid(avg_out + max_out)
        spectral_out = spectral_weight * x
        spatial_avg = torch.mean(spectral_out, dim=1, keepdim=True)
        spatial_max, _ = torch.max(spectral_out, dim=1, keepdim=True)
        spatial_pool = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_pool))
        return spatial_weight * spectral_out



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
            return torch.cat([primary, self.ghost_conv(primary)], dim=1)
        return primary



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



class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super(ConvNeXtBlock, self).__init__()
        hidden = out_ch * expand_ratio
        self.dwconv = nn.Conv2d(in_ch, in_ch, 7, stride, 3, groups=in_ch, bias=False)
        self.norm = nn.GroupNorm(min(in_ch // 8, 32), in_ch)
        self.pwconv1 = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, out_ch, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.GroupNorm(min(out_ch // 8, 32), out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return out + identity


class EnhancedSSA(nn.Module):
    def __init__(self, channels, reduction=16, groups=4):
        super(EnhancedSSA, self).__init__()
        self.groups = groups
        ch_per_group = channels // groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_per_group, ch_per_group // reduction, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(ch_per_group // reduction, ch_per_group, 1, bias=False)
            ) for _ in range(groups)
        ])
        self.spatial_dilate1 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, groups=2, bias=False)
        self.spatial_dilate2 = nn.Conv2d(2, 2, 3, padding=5, dilation=5, groups=2, bias=False)
        self.spatial_fuse = nn.Conv2d(6, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        ch_per = C // self.groups
        spe_weights = []
        for g in range(self.groups):
            gx = x[:, g * ch_per:(g + 1) * ch_per, :, :]
            avg_out = self.group_mlps[g](self.avg_pool(gx))
            max_out = self.group_mlps[g](self.max_pool(gx))
            spe_weights.append(self.sigmoid(avg_out + max_out))
        spectral_weight = torch.cat(spe_weights, dim=1)
        spectral_out = spectral_weight * x
        spatial_avg = torch.mean(spectral_out, dim=1, keepdim=True)
        spatial_max, _ = torch.max(spectral_out, dim=1, keepdim=True)
        pool_base = torch.cat([spatial_avg, spatial_max], dim=1)
        pool_d3 = self.spatial_dilate1(pool_base)
        pool_d5 = self.spatial_dilate2(pool_base)
        pool_cat = torch.cat([pool_base, pool_d3, pool_d5], dim=1)
        spatial_weight = self.sigmoid(self.spatial_fuse(pool_cat))
        return spatial_weight * spectral_out


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv_d6 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv_d9 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=9, dilation=9, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        f1 = self.conv1x1(x)
        f3 = self.conv_d3(x)
        f6 = self.conv_d6(x)
        f9 = self.conv_d9(x)
        gp = F.interpolate(self.global_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([f1, f3, f6, f9, gp], dim=1))



class UniversalEncoderV7(nn.Module):
    def __init__(self, in_channels=110):
        super(UniversalEncoderV7, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.down1 = ConvNeXtBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvNeXtBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvNeXtBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck_conv = ConvNeXtBlock(256, 512)
        self.bottleneck_aspp = ASPP(512, 512)
        self.bottleneck_att = EnhancedSSA(512)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck_conv(self.pool3(x3))
        x4 = self.bottleneck_aspp(x4)
        x4 = self.bottleneck_att(x4)
        return x1, x2, x3, x4



class DiseaseSpecificDecoderV7(nn.Module):
    def __init__(self, out_channels=1):
        super(DiseaseSpecificDecoderV7, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = SpectralSpatialAttention(256)
        self.up_conv1 = GhostResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = SpectralSpatialAttention(128)
        self.up_conv2 = GhostResBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = SpectralSpatialAttention(64)
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



class STARNetH(nn.Module):
    def __init__(self, in_channels=110, tasks=['grape', 'corn', 'tomato']):
        super(STARNetH, self).__init__()
        self.tasks = tasks
        self.encoder = UniversalEncoderV7(in_channels)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, len(tasks))
        )
        self.decoders = nn.ModuleDict({
            task: DiseaseSpecificDecoderV7() for task in tasks
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
