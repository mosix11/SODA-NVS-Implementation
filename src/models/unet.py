

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import TimeEmbedding, LatentEmbedding, PoseEmbedding
from .blocks import Upsample, Downsample, ResidualBlock
from .blocks import DownsampleResBlock, UpsampleResBlock, ResAttBlock, MiddleBlock

class UNet(nn.Module):
    def __init__(self, 
                 image_shape = [3, 64, 64], 
                 n_channels = 128,
                 ch_mults = (1, 2, 2, 2),
                 is_attn = (False, True, False, False),
                 attn_channels_per_head = None,
                 dropout = 0.1,
                 n_blocks = 2,
                 use_res_for_updown = False,
                 time_embed_dim = 512,
                 z_channels = 128,
                 grid_condition_dim = None,
                 self_attention_type = 'nromal'
                 ):
        """
        * `image_shape` is the (channel, height, width) size of images.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `n_channels * ch_mults[i]`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `dropout` is the dropout rate
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        * `time_embed_dim` the dimension of the time embedding vector
        * `use_res_for_updown` indicates whether to use ResBlocks for up/down sampling (BigGAN-style)
        * `z_channels` is the number channels in the latent code derived by the resnet encoder
        * `grid_condition_dim` is the number of channels of the H×W 2D conditioning grid (for novel view synthesis)
        * `self_attention_type` is the method used for spatial self attention operation (normal, torch, flash)
        """
        super().__init__()
        
        n_resolutions = len(ch_mults)
        
        
        if grid_condition_dim:
            self.grid_cond_emb = PoseEmbedding(num_freqs=20)
            # Linear
            self.rgb_linear_projector = nn.Conv2d(image_shape[0], n_channels, kernel_size=1)
            self.image_proj = nn.Conv2d(image_shape[0] + grid_condition_dim, n_channels, kernel_size=3, padding=1)
        else:
            self.image_proj = nn.Conv2d(image_shape[0], n_channels, kernel_size=3, padding=1)
        
        # Time embedding layer.
        # time_channels = n_channels * 4
        time_channels = time_embed_dim
        self.time_emb = TimeEmbedding(time_channels)
        
        # Latent embedding layer.
        self.z_emb = LatentEmbedding(z_channels)
        
        
        # Down stages
        down = []
        in_channels = n_channels
        h_channels = [n_channels]
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks` at the same resolution
            down.append(ResAttBlock(in_channels, out_channels, time_channels, z_channels, is_attn[i], attn_channels_per_head, attn_type=self_attention_type, dropout=dropout))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 1):
                down.append(ResAttBlock(out_channels, out_channels, time_channels, z_channels, is_attn[i], attn_channels_per_head, attn_type=self_attention_type, dropout=dropout))
                h_channels.append(out_channels)
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                if use_res_for_updown:
                    down.append(DownsampleResBlock(out_channels, time_channels, z_channels, dropout))
                else:
                    down.append(Downsample(out_channels))
                h_channels.append(out_channels)
            in_channels = out_channels
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, z_channels, attn_channels_per_head, attn_type=self_attention_type, dropout=dropout)

        # Up stages
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks + 1` at the same resolution
            for _ in range(n_blocks + 1):
                up.append(ResAttBlock(in_channels + h_channels.pop(), out_channels, time_channels, z_channels, is_attn[i], attn_channels_per_head, attn_type=self_attention_type, dropout=dropout))
                in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                if use_res_for_updown:
                    up.append(UpsampleResBlock(out_channels, time_channels, z_channels, attn_type=self_attention_type, dropout=dropout))
                else:
                    up.append(Upsample(out_channels))
        assert not h_channels
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(out_channels, image_shape[0], kernel_size=3, padding=1)
        
        
    def forward(self, x, t, z, drop_mask, grid_condition=None, ret_activation=False):
        if not ret_activation:
            return self.forward_core(x, t, z, drop_mask)

        activation = {}
        def namedHook(name):
            def hook(module, input, output):
                activation[name] = output
            return hook
        hooks = {}
        no = 0
        for blk in self.up:
            if isinstance(blk, ResAttBlock):
                no += 1
                name = f'out_{no}'
                hooks[name] = blk.register_forward_hook(namedHook(name))

        result = self.forward_core(x, t, z, drop_mask, grid_condition)
        for name in hooks:
            hooks[name].remove()
        return result, activation

    def forward_core(self, x, t, z, drop_mask, grid_condition=None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        * `z` has shape `[batch_size, z_channels]`
        * `drop_mask` has shape `[batch_size]`
        * `grid_condition` has shape `[batch_size, ray_dim, height, width]`
        """

        t = self.time_emb(t)
        z = self.z_emb(z, drop_mask)
        if grid_condition:
            c = self.grid_cond_emb(grid_condition)
            x = self.rgb_linear_projector(x)
            x = self.image_proj(torch.cat((x, c), dim=1))
        else:
            x = self.image_proj(x)
        
        # `h` will store outputs at each resolution for skip connection
        h = [x]

        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            elif isinstance(m, DownsampleResBlock):
                x = m(x, t, z)
            else:
                x = m(x, t, z).contiguous()
            h.append(x)

        x = self.middle(x, t, z).contiguous()

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            elif isinstance(m, UpsampleResBlock):
                x = m(x, t, z)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, z).contiguous()

        return self.final(self.act(self.norm(x)))