import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.silu(h)

        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.silu(h)
        h = self.dropout(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v  # (B, heads, HW, C//heads)

        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
        use_attention=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        channels = [base_channels]
        in_ch = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, dropout))
                if use_attention and i >= len(channel_mults) - 2:  # Add attention at lower resolutions
                    blocks.append(AttentionBlock(out_ch))
                in_ch = out_ch

            self.encoder_blocks.append(blocks)
            self.encoder_downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            channels.append(out_ch)

        # Middle
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(channels[-1], channels[-1], time_emb_dim, dropout),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1], time_emb_dim, dropout),
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        in_ch = channels[-1]

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = channels[-(i + 2)]

            self.skip_convs.append(nn.Conv2d(skip_ch, skip_ch, 1))

            if i < len(channel_mults) - 1:
                self.decoder_ups.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            else:
                if in_ch != out_ch:
                    self.decoder_ups.append(nn.Conv2d(in_ch, out_ch, 1))
                else:
                    self.decoder_ups.append(nn.Identity())

            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                if j == 0:
                    block_in_ch = out_ch + skip_ch
                else:
                    block_in_ch = out_ch
                blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))
                if use_attention and i <= 1:  # Add attention at higher resolutions
                    blocks.append(AttentionBlock(out_ch))

            self.decoder_blocks.append(blocks)

            in_ch = out_ch

        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):

        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.init_conv(x)

        # Encoder
        skips = [h]
        for blocks, down in zip(self.encoder_blocks, self.encoder_downs):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            skips.append(h)
            h = down(h)

        # Middle
        for block in self.middle_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Decoder
        skips = skips[:-1][::-1]
        for blocks, up, skip_conv, skip in zip(self.decoder_blocks, self.decoder_ups, self.skip_convs, skips):
            h = up(h)
            skip = skip_conv(skip)

            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)

            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        h = torch.tanh(h) * 0.5 + 0.5
        return h


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        # Middle
        self.mid = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        # Output
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)

        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)

        h = self.mid[0](h3, t_emb)
        h = self.mid[1](h, t_emb)

        h = self.dec3(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h)

        return self.out(h)
