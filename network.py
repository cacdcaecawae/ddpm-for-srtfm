from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def map_timestep(t: torch.Tensor,
                 noise_levels: Optional[torch.Tensor]) -> torch.Tensor:
    """
    将离散时间步索引映射为连续 noise level（如果提供），否则保持原样。
    返回浮点张量以支持正弦时间编码。
    """
    if noise_levels is None:
        return t.float()
    if t.dtype.is_floating_point:
        return t
    return noise_levels.to(t.device)[t]


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal encoding.")
        self.dim = d_model
        self.half = d_model // 2
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        连续/离散时间步的正弦位置编码 (支持浮点时间)。
        """
        device = t.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=self.half, dtype=torch.float32, device=device)
            / self.half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.activation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(inputs)
        x = self.activation2(x)
        return x


class ConvNet(nn.Module):

    def __init__(self,
                 n_steps: int,
                 in_channels: int = 1,
                 intermediate_channels=None,
                 pe_dim: int = 10,
                 insert_t_to_all_layers: bool = False,
                 noise_levels: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = [10, 20, 40]
        self.pe = PositionalEncoding(n_steps, pe_dim)
        if noise_levels is not None:
            noise_levels = noise_levels.float()
        self.register_buffer("noise_levels", noise_levels, persistent=False)

        self.pe_linears = nn.ModuleList()
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, in_channels))

        self.residual_blocks = nn.ModuleList()
        prev_channel = in_channels
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = t.shape[0]
        t = map_timestep(t, self.noise_levels)
        t = self.pe(t)
        for block, linear in zip(self.residual_blocks, self.pe_linears):
            if linear is not None:
                pe = linear(t).reshape(batch_size, -1, 1, 1)
                x = x + pe
            x = block(x)
        x = self.output_layer(x)
        return x


class SeqT(nn.Module):

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for module in self.mods:
            x = module(x, t_emb)
        return x


class UnetBlock(nn.Module):

    def __init__(self,
                 shape: tuple[int, int, int],
                 in_c: int,
                 out_c: int,
                 t_dim: Optional[int] = None,
                 residual: bool = False) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)
        else:
            self.residual_conv = None

        self.t_dim = t_dim
        if t_dim is not None:
            self.t_fc1 = nn.Linear(t_dim, out_c)
            self.t_fc2 = nn.Linear(t_dim, out_c)
        else:
            self.t_fc1 = None
            self.t_fc2 = None

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        if t_emb is not None:
            t_emb = t_emb.squeeze(1)
        if self.t_fc1 is not None and t_emb is not None:
            out = out + self.t_fc1(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(out)
        if self.t_fc2 is not None and t_emb is not None:
            out = out + self.t_fc2(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        if self.residual:
            out = out + self.residual_conv(x)
        out = self.activation(out)
        return out


class UNet(nn.Module):

    def __init__(self,
                 n_steps: int,
                 in_channels: int = 1,
                 image_size: int = 101,
                 lr_channels: Optional[int] = None,
                 channels=None,
                 pe_dim: int = 128,
                 residual: bool = False,
                 noise_levels: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if channels is None:
            channels = [10, 20, 40, 80]
        c, h, w = in_channels, image_size, image_size
        
        # LR 通道数默认与 HR 相同,但在 TFM 模式下可能不同
        if lr_channels is None:
            lr_channels = in_channels
        
        self.in_channels = in_channels
        self.lr_channels = lr_channels
        if noise_levels is not None:
            noise_levels = noise_levels.float()
        self.register_buffer("noise_levels", noise_levels, persistent=False)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(pe_dim, pe_dim * 4), nn.SiLU(),
            nn.Linear(pe_dim * 4, pe_dim * 4))
        self.t_dim = pe_dim

        num_layers = len(channels)
        hs = [h]
        ws = [w]
        current_h, current_w = h, w
        for _ in range(num_layers - 1):
            current_h //= 2
            current_w //= 2
            hs.append(current_h)
            ws.append(current_w)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 输入是 HR + LR concatenated
        prev_channel = c + lr_channels
        for channel, ch, cw in zip(channels[:-1], hs[:-1], ws[:-1]):
            self.encoders.append(
                SeqT(
                    UnetBlock((prev_channel, ch, cw),
                              prev_channel,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                    UnetBlock((channel, ch, cw),
                              channel,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                ))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        mid_channel = channels[-1]
        self.mid = SeqT(
            UnetBlock((prev_channel, hs[-1], ws[-1]),
                      prev_channel,
                      mid_channel,
                      t_dim=self.t_dim,
                      residual=residual),
            UnetBlock((mid_channel, hs[-1], ws[-1]),
                      mid_channel,
                      mid_channel,
                      t_dim=self.t_dim,
                      residual=residual),
        )
        prev_channel = mid_channel

        for channel, ch, cw in zip(channels[-2::-1], hs[-2::-1], ws[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                SeqT(
                    UnetBlock((channel * 2, ch, cw),
                              channel * 2,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                    UnetBlock((channel, ch, cw),
                              channel,
                              channel,
                              t_dim=self.t_dim,
                              residual=residual),
                ))
            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, c, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                lr_image: torch.Tensor) -> torch.Tensor:
        t = map_timestep(t, self.noise_levels)
        x = torch.cat((x, lr_image), dim=1)
        t_emb = self.pe(t)
        encoder_outs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, t_emb)
            encoder_outs.append(x)
            x = down(x)
        x = self.mid(x, t_emb)
        for decoder, up, encoder_out in zip(self.decoders, self.ups,
                                            encoder_outs[::-1]):
            x = up(x)
            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x,
                      (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                       pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x, t_emb)
        return self.conv_out(x)


class AdaLayerNorm(nn.Module):

    def __init__(self, normalized_shape: int, t_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, normalized_shape * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mlp(t_emb).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 t_dim: int = 256) -> None:
        super().__init__()
        self.adaln_attn = AdaLayerNorm(embed_dim, t_dim)
        self.attn = nn.MultiheadAttention(embed_dim,
                                          num_heads,
                                          dropout=attn_dropout,
                                          batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        self.adaln_mlp = AdaLayerNorm(embed_dim, t_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.adaln_attn(x, t_emb)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(attn_out)
        h = self.adaln_mlp(x, t_emb)
        x = x + self.mlp(h)
        return x


class DiT(nn.Module):

    def __init__(self,
                 n_steps: int,
                 in_channels: int = 1,
                 lr_channels: Optional[int] = None,
                 patch_size: int = 4,
                 embed_dim: int = 256,
                 depth: int = 8,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 pe_dim: int = 256,
                 noise_levels: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if lr_channels is None:
            lr_channels = in_channels
        if embed_dim % 4 != 0:
            raise ValueError(
                "embed_dim must be divisible by 4 for 2D sin-cos positional encoding.")

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.lr_channels = lr_channels
        self.embed_dim = embed_dim
        if noise_levels is not None:
            noise_levels = noise_levels.float()
        self.register_buffer("noise_levels", noise_levels, persistent=False)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(pe_dim, pe_dim * 4), nn.SiLU(),
            nn.Linear(pe_dim * 4, embed_dim))

        self.patch_embed = nn.Conv2d(in_channels + lr_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim=embed_dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     dropout=dropout,
                     attn_dropout=attn_dropout,
                     t_dim=embed_dim) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatch = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels * (patch_size ** 2), 1),
            nn.PixelShuffle(patch_size),
        )
        self.refine = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _build_2d_sincos_position_embedding(self, h: int, w: int,
                                            device: torch.device,
                                            dtype: torch.dtype) -> torch.Tensor:
        grid_w = torch.arange(w, device=device, dtype=dtype)
        grid_h = torch.arange(h, device=device, dtype=dtype)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=dtype)
        omega = 1.0 / (10000**(omega / pos_dim))

        outs = []
        for g in grid:
            g_flat = g.reshape(-1)
            outs.append(torch.sin(torch.outer(g_flat, omega)))
            outs.append(torch.cos(torch.outer(g_flat, omega)))
        pos_embed = torch.cat(outs, dim=1)
        return pos_embed.unsqueeze(0)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                lr_image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        t = map_timestep(t, self.noise_levels)
        x = torch.cat((x, lr_image), dim=1)
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.patch_embed(x)
        h_p, w_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        pos_embed = self._build_2d_sincos_position_embedding(
            h_p, w_p, x.device, x.dtype)
        x = self.pos_drop(x + pos_embed)

        t_emb = self.t_mlp(self.pe(t))
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.final_norm(x)

        x = x.transpose(1, 2).reshape(b, self.embed_dim, h_p, w_p)
        x = self.unpatch(x)
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        x = x + self.refine(x)
        return x


convnet_small_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 20],
    'pe_dim': 128,
}

convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True,
}

convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True,
}

unet_1_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}

unet_res_cfg = {
    'type': 'UNet',
    'channels': [16, 32, 64, 128, 256],
    'pe_dim': 128,
    'residual': True,
}

dit_base_cfg = {
    'type': 'DiT',
    'patch_size': 8,
    'embed_dim': 256,
    'depth': 6,
    'num_heads': 4,
    'mlp_ratio': 4.0,
    'dropout': 0.0,
    'attn_dropout': 0.0,
    'pe_dim': 256,
}


def build_network(config: dict, n_steps: int, in_channels: int = 1, image_size: int = 101, lr_channels: Optional[int] = None, noise_levels: Optional[torch.Tensor] = None) -> nn.Module:
    cfg = config.copy()
    network_type = cfg.pop('type')
    
    # 添加 in_channels 和 image_size 到配置中
    cfg['in_channels'] = in_channels
    if network_type == 'UNet':
        cfg['image_size'] = image_size
    if network_type in ('UNet', 'DiT') and lr_channels is not None:
        cfg['lr_channels'] = lr_channels
    if noise_levels is not None:
        cfg['noise_levels'] = noise_levels
    
    if network_type == 'ConvNet':
        network_cls = ConvNet
    elif network_type == 'UNet':
        network_cls = UNet
    elif network_type == 'DiT':
        network_cls = DiT
    else:
        raise KeyError(f"Unsupported network type '{network_type}'.")
    return network_cls(n_steps, **cfg)

