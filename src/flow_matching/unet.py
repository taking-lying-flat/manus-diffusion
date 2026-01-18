import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads
        scale = 1 / math.sqrt(head_dim)
        q = q.reshape(b * self.num_heads, head_dim, h * w).transpose(1, 2)
        k = k.reshape(b * self.num_heads, head_dim, h * w)
        v = v.reshape(b * self.num_heads, head_dim, h * w).transpose(1, 2)
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        h_out = torch.bmm(attn, v)
        h_out = h_out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_out(h_out.reshape(b, c, -1)).reshape(b, c, h, w)


class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        num_heads=1,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, model_channels, 3, padding=1)]
        )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = nn.ModuleList(
            [
                ResBlock(ch, time_embed_dim, dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResBlock(ch, time_embed_dim, dropout),
            ]
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            hs.append(h)
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
        return self.out(h)

