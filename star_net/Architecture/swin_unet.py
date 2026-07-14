import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 核心组件 1：窗口划分与还原
# ==========================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ==========================================================
# 核心组件 2：多头自注意力 (带相对位置编码)
# ==========================================
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==========================================================
# 核心组件 3：Swin Transformer Block
# ==========================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4))

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ==========================================================
# 核心组件 4：Patch 操作 (降采样与升采样)
# ==========================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=110, embed_dim=96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, 2).permute(0, 2, 3, 1).view(B, -1, C // 4)
        return self.norm(x)


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 16 * dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, int(L ** 0.5), int(L ** 0.5), -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, 4)
        return x


# ==========================================================
# 最终架构：Swin-Unet
# ==========================================================
class SwinUNet(nn.Module):
    """
    Swin-Unet 官方拓扑结构 (Modified for HSI)
    - 纯 Transformer 编码器-解码器架构
    - 原生支持 110 波段输入
    """

    def __init__(self, in_channels=110, out_channels=1, img_size=256, embed_dim=96, depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24], window_size=8):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=4, in_chans=in_channels, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution

        # Encoder
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.ModuleList([
                BasicLayer(dim=int(embed_dim * 2 ** i),
                           input_resolution=(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)),
                           depth=depths[i], num_heads=num_heads[i], window_size=window_size),
                PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)),
                             dim=int(embed_dim * 2 ** i)) if (i < self.num_layers - 1) else nn.Identity()
            ])
            self.layers.append(layer)

        # Decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_layers - 1):
            res_idx = self.num_layers - 1 - i
            res_h = patches_resolution[0] // (2 ** res_idx)

            # PatchExpand 进行上采样
            self.layers_up.append(PatchExpand(input_resolution=(res_h, res_h), dim=int(embed_dim * 2 ** res_idx)))

            # 跳跃连接后的通道融合 (C + C/2 -> C/2)
            self.concat_back_dim.append(nn.Linear(int(embed_dim * 2 ** res_idx),
                                                  int(embed_dim * 2 ** (res_idx - 1))))

            # BasicLayer 进行特征解码
            self.layers_up.append(
                BasicLayer(dim=int(embed_dim * 2 ** (res_idx - 1)), input_resolution=(res_h * 2, res_h * 2),
                           depth=depths[res_idx - 1], num_heads=num_heads[res_idx - 1], window_size=window_size))

        # Final Expansion
        self.final_up = FinalPatchExpand_X4(input_resolution=patches_resolution, dim=embed_dim)
        self.output = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)

        # 编码阶段保存跳跃连接特征
        skip_features = []
        for i, (basic_layer, merging) in enumerate(self.layers):
            x = basic_layer(x)
            if i < self.num_layers - 1:
                skip_features.append(x)
                x = merging(x)

        # 解码阶段
        for i in range(self.num_layers - 1):
            idx = self.num_layers - 2 - i
            expand_layer = self.layers_up[i * 2]
            concat_layer = self.concat_back_dim[i]
            basic_layer_up = self.layers_up[i * 2 + 1]

            x_up = expand_layer(x)
            x_skip = skip_features[idx]

            # 拼接并融合维度
            x = torch.cat([x_up, x_skip], dim=-1)
            x = concat_layer(x)
            x = basic_layer_up(x)

        # 还原至原图大小
        x = self.final_up(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    # 🌟 传入 window_size=8 以完美适配 256 的图像分辨率
    model = SwinUNet(in_channels=110, out_channels=1, img_size=256, window_size=8)
    dummy_input = torch.randn(2, 110, 256, 256)
    output = model(dummy_input)
    print(f"✅ Swin-Unet 测试通过！输入: {dummy_input.shape} -> 输出: {output.shape}")