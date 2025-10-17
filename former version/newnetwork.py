import torch
import torch.nn as nn
import torch.nn.functional as F

from dldemos.ddpm.dataset import get_img_shape


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.actvation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)
        x = self.actvation2(x)
        return x


class ConvNet(nn.Module):

    def __init__(self,
                 n_steps,
                 intermediate_channels=[10, 20, 40],
                 pe_dim=10,
                 insert_t_to_all_layers=False):
        super().__init__()
        C, H, W = get_img_shape()  # 1, 28, 28
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))

        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                pe = m_t(t).reshape(n, -1, 1, 1)
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x

class SeqT(nn.Module):
    def __init__(self, *mods):
        super().__init__()
        self.mods = nn.ModuleList(mods)
    def forward(self, x, t_emb):
        for m in self.mods:
            # 这里假定子模块都接受 (x, t_emb)
            x = m(x, t_emb)
        return x

class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, t_dim=None, residual=False):
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

        # 新增：时间线性层（两处注入，conv1后/conv2后各一次，更稳）
        self.t_dim = t_dim
        if t_dim is not None:
            self.t_fc1 = nn.Linear(t_dim, out_c)
            self.t_fc2 = nn.Linear(t_dim, out_c)

    def forward(self, x, t_emb=None):
        # x: (B, C, H, W); t_emb: (B, t_dim)
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        t_emb = t_emb.squeeze(1)
        if (self.t_dim is not None) and (t_emb is not None):
            out = out + self.t_fc1(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)

        out = self.conv2(out)

        if (self.t_dim is not None) and (t_emb is not None):
            out = out + self.t_fc2(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)

        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out



class UNet(nn.Module):
    def __init__(self, n_steps, channels=[10,20,40,80], pe_dim=128, residual=False):
        super().__init__()
        C, H, W = get_img_shape()

        # 1) 时间编码 + MLP 放大维度
        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(pe_dim, pe_dim*4), nn.SiLU(),
            nn.Linear(pe_dim*4, pe_dim*4),
        )
        self.t_dim = pe_dim*1

        # ↓↓↓ 原来计算 Hs/Ws 的代码保持不变 ↓↓↓
        layers = len(channels)
        Hs = [H]; Ws = [W]; cH, cW = H, W
        for _ in range(layers - 1):
            cH //= 2; cW //= 2
            Hs.append(cH); Ws.append(cW)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.ups      = nn.ModuleList()

        prev_channel = C*2  # x 与 lr_image concat
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            # 2) 用 SeqT + UnetBlock(t_dim=...) 替换 nn.Sequential
            self.encoders.append(
                SeqT(
                    UnetBlock((prev_channel, cH, cW), prev_channel, channel, t_dim=self.t_dim, residual=residual),
                    UnetBlock((channel,     cH, cW), channel,     channel, t_dim=self.t_dim, residual=residual),
                )
            )
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        # mid
        channel = channels[-1]
        self.mid = SeqT(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]), prev_channel, channel, t_dim=self.t_dim, residual=residual),
            UnetBlock((channel,      Hs[-1], Ws[-1]), channel,      channel, t_dim=self.t_dim, residual=residual),
        )
        prev_channel = channel

        # decoders
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                SeqT(
                    UnetBlock((channel*2, cH, cW), channel*2, channel, t_dim=self.t_dim, residual=residual),
                    UnetBlock((channel,  cH, cW), channel,    channel, t_dim=self.t_dim, residual=residual),
                )
            )
            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t, lr_image):
        x = torch.cat((x, lr_image), dim=1)
        t_emb = self.pe(t) # (B, t_dim)
        encoder_outs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x, t_emb)
            encoder_outs.append(x)
            x = down(x)

        x = self.mid(x, t_emb)

        for dec, up, encoder_out in zip(self.decoders, self.ups, encoder_outs[::-1]):
            x = up(x)
            # 尺寸对齐（和你原来一致）
            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = dec(x, t_emb)

        return self.conv_out(x)


convnet_small_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 20],
    'pe_dim': 128
}

convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}
convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

unet_1_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
unet_res_cfg = {
    'type': 'UNet',
    'channels': [16, 32, 64, 128, 256],
    'pe_dim': 128,
    'residual': True
}


def build_network(config: dict, n_steps):
    network_type = config.pop('type')
    if network_type == 'ConvNet':
        network_cls = ConvNet
    elif network_type == 'UNet':
        network_cls = UNet

    network = network_cls(n_steps, **config)
    return network
