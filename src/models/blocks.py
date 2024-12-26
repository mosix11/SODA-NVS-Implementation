
import torch
from torch import nn
import torch.nn.functional as F

from flash_attn import flash_attn_qkvpacked_func



from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import os

def GroupNorm32(channels):
    return nn.GroupNorm(32, channels, affine=False)


class Upsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest'),
        if use_conv:
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up(x)
        if self.use_conv:
            return self.conv(x)
        else:
            return x


class Downsample(nn.Module):
    def __init__(self, n_channels, use_conv=True, rearrange_pooling=False):
        super().__init__()
        self.use_conv = use_conv
        self.rearrange_pooling = rearrange_pooling
        if use_conv:
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)
        elif rearrange_pooling:
            self.rarng_pool = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
                nn.Conv2d(n_channels * 4, n_channels, kernel_size=1)
            )
        else:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        elif self.rearrange_pooling:
            return self.rarng_pool(x)
        else:
            return self.pool(x)




class AttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k

        self.norm = GroupNorm32(n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = 1 / torch.sqrt(torch.sqrt(d_k))
        self.n_heads = n_heads
        self.d_k = d_k
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            print(f"{self.n_heads} heads, {self.d_k} channels per head")

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """
        batch_size, n_channels, height, width = x.shape
        # Normalize and rearrange to `[batch_size, seq, n_channels]`
        h = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)

        # {q, k, v} all have a shape of `[batch_size, seq, n_heads, d_k]`
        qkv = self.projection(h).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q * self.scale, k * self.scale) # More stable with f16 than dividing afterwards
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape to `[batch_size, seq, n_heads * d_k]` and transform to `[batch_size, seq, n_channels]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res + x


class TorchAttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k):
        """
        * `n_channels` is the number of channels in the input
        * `d_k` is the number of dimensions in each head
        """
        super().__init__()
        
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k

        self.norm = GroupNorm32(n_channels)

        # PyTorch MultiheadAttention module
        self.attention = nn.MultiheadAttention(embed_dim=n_channels, num_heads=n_heads, batch_first=True)

        self.d_k = d_k
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            print(f"{n_heads} heads, {d_k} channels per head")

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """
        batch_size, n_channels, height, width = x.shape
        
        # Normalize and rearrange to `[batch_size, seq_len, n_channels]`
        h = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)

        # MultiheadAttention expects `[seq_len, batch_size, n_channels]` if batch_first=False
        h = h.permute(1, 0, 2)  # `[seq_len, batch_size, n_channels]`
        attn_output, _ = self.attention(h, h, h)

        # Reshape back to `[batch_size, n_channels, height, width]`
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, n_channels, height, width)
        return attn_output + x

class FlashAttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k=None):
        """
        * `n_channels` is the number of channels in the input
        * `d_k` is the number of dimensions in each head (head_dim in FlashAttention)
        """
        super().__init__()
        
        # Set default `d_k` if None
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k

        self.norm = GroupNorm32(n_channels)
        self.projection = nn.Linear(n_channels, 3 * n_channels)  # For Q, K, V

        self.n_heads = n_heads
        self.d_k = d_k

        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            print(f"{n_heads} heads, {d_k} channels per head")

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """
        batch_size, n_channels, height, width = x.shape

        # Normalize and rearrange to `[batch_size, seq_len, n_channels]`
        h = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)

        # Project input to Q, K, V packed into one tensor
        qkv = self.projection(h).view(batch_size, -1, self.n_heads, 3, self.d_k)
        qkv = qkv.permute(0, 1, 3, 2, 4)  # `[batch_size, seq_len, 3, n_heads, d_k]`

        # Apply FlashAttention
        attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=0.0)

        # Reshape back to `[batch_size, seq_len, n_channels]`
        attn_output = attn_output.reshape(batch_size, -1, n_channels)

        # Convert back to `[batch_size, n_channels, height, width]`
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return attn_output + x


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        # Create sinusoidal position embeddings (same as those from the transformer)
        half_dim = self.n_channels // 8
        emb = torch.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb
    
    
class LatentEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels

    def forward(self, z, drop_mask):
        """
        * `z` is the latent code
        * `drop_mask`: mask out the condition if drop_mask == 1
        """
        drop_mask = drop_mask[:, None]
        drop_mask = drop_mask.repeat(1, self.n_channels)
        drop_mask = 1 - drop_mask  # need to flip 0 <-> 1
        z = z * drop_mask
        return z


class PoseEmbedding(nn.Module):
    def __init__(self,
                 scale:float = 1e-4,
                 num_freqs:int = 10 
                ):
        """
        NeRF style postional encoding for wrold positions and directions
        """
        super().__init__()

        self.scale = scale
        self.num_freqs = num_freqs
        
    
    def forward(self, c):
        
        c = self.pose_sinusoidal_encoding(c)
        
        return c

    def pose_sinusoidal_encoding(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        SODA-style sinusoidal encoding for rays (or any coordinates).

        Args:
        x:          (..., D) tensor of coordinates; D can be 2, 3, 6, etc.
        num_freqs:  number of frequency bands (L) for the encoding
        scale:      's' hyperparameter, e.g. 0.0001 as in Table 12 of the paper

        Returns:
        A tensor of shape (..., 2 * D * num_freqs), containing:
            [
            sin(1 * 2π * s * x), cos(1 * 2π * s * x),
            sin(2 * 2π * s * x), cos(2 * 2π * s * x),
            ...
            up to 2^(num_freqs - 1)
            ]
        """
        
        num_freqs = self.num_freqs
        scale = self.scale
        
        # 1) Remember the original shape so we can restore it later
        orig_shape = x.shape
        D = orig_shape[-1]  # last dimension is the coordinate dimension

        # 2) Flatten everything except the last dimension
        x_flat = x.view(-1, D)  # shape (N, D)
        n = x_flat.shape[0]     # number of flattened elements

        # 3) Create frequency multipliers [1, 2, 4, ..., 2^(num_freqs-1)]
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device=x.device)

        # 4) Scale the input by (2π * s) and then multiply by each frequency
        scaled_x = (2.0 * torch.pi * scale) * x_flat.unsqueeze(-1)  # (N, D, 1)
        scaled_x = scaled_x * freqs.view(1, 1, num_freqs)          # (N, D, num_freqs)

        # 5) Apply sin and cos
        sin_x = torch.sin(scaled_x)  # (N, D, num_freqs)
        cos_x = torch.cos(scaled_x)  # (N, D, num_freqs)

        # 6) Concatenate sin and cos => shape (N, D, 2*num_freqs)
        enc = torch.cat([sin_x, cos_x], dim=-1)

        # 7) Flatten out the D dimension => shape (N, 2*D*num_freqs)
        enc = enc.view(n, -1)

        # 8) Reshape back to (..., 2*D*num_freqs)
        out_shape = list(orig_shape[:-1]) + [2 * D * num_freqs]
        enc = enc.view(*out_shape)

        return enc
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, z_channels, dropout=0.1, up=False, down=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `z_channels` is the number channels in the latent code derived by the resnet encoder
        * `dropout` is the dropout rate
        """
        super().__init__()
        self.norm1 = GroupNorm32(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = GroupNorm32(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for embeddings
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, 2 * out_channels)
        )
        self.z_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_channels, 2 * out_channels)
        )

        # BigGAN style: use resblock for up/downsampling
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, use_conv=False)
            self.x_upd = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

    def forward(self, x, t, z):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        * `z` has shape `[batch_size, z_channels]`
        """
        if self.updown:
            h = self.norm2(self.conv1(self.h_upd(self.act1(self.norm1(x)))))
            x = self.x_upd(x)
        else:
            h = self.norm2(self.conv1(self.act1(self.norm1(x))))

        # TODO condition on z using cross attention
        # Adaptive Group Normalization
        t_s, t_b = self.time_emb(t).chunk(2, dim=1)
        z_s, z_b = self.z_emb(z).chunk(2, dim=1)
        h = t_s[:, :, None, None] * h + t_b[:, :, None, None]
        h = z_s[:, :, None, None] * h + z_b[:, :, None, None]

        h = self.conv2(self.act2(h))
        return h + self.shortcut(x)
    
    
class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, z_channels, has_attn, attn_channels_per_head, attn_type='normal', dropout=0.1):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, z_channels, dropout=dropout)
        if has_attn:
            if attn_type == 'normal':
                self.attn = AttentionBlock(out_channels, attn_channels_per_head)
            elif attn_type == 'torch':
                self.attn = TorchAttentionBlock(out_channels, attn_channels_per_head)
            elif attn_type == 'flash':
                self.attn = FlashAttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t, z):
        x = self.res(x, t, z)
        x = self.attn(x)
        return x
    
    
class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, z_channels, attn_channels_per_head, attn_type='normal', dropout=0.1):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout)
        if attn_type == 'normal':
            self.attn = AttentionBlock(n_channels, attn_channels_per_head)
        elif attn_type == 'torch':
            self.attn = TorchAttentionBlock(n_channels, attn_channels_per_head)
        elif attn_type == 'flash':
            self.attn = FlashAttentionBlock(n_channels, attn_channels_per_head)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout)

    def forward(self, x, t, z):
        x = self.res1(x, t, z)
        x = self.attn(x)
        x = self.res2(x, t, z)
        return x
    
class UpsampleResBlock(nn.Module):
    def __init__(self, n_channels, time_channels, z_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout, up=True)

    def forward(self, x, t, z):
        return self.op(x, t, z)
    
class DownsampleResBlock(nn.Module):
    def __init__(self, n_channels, time_channels, z_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, z_channels, dropout=dropout, down=True)

    def forward(self, x, t, z):
        return self.op(x, t, z)