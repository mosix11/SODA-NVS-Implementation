import torch
from torch import nn
import torch.nn.functional as F

from .blocks import TimeEmbedding, PoseEmbedding
from .blocks import Upsample, Downsample
from .blocks import AttentionBlock, TorchAttentionBlock, FlashAttentionBlock
from .blocks import DownsampleResBlock, UpsampleResBlock, ResAttBlock, MiddleBlock



  
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1, up=False, down=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `dropout` is the dropout rate
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, affine=False)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels, affine=False)
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

    def forward(self, x, t):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        if self.updown:
            h = self.norm2(self.conv1(self.h_upd(self.act1(self.norm1(x)))))
            x = self.x_upd(x)
        else:
            h = self.norm2(self.conv1(self.act1(self.norm1(x))))
        h = self.conv2(self.act2(h))
        return h + self.shortcut(x)
    
    
class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn, attn_channels_per_head, attn_type='normal', dropout=0.1):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, dropout=dropout)
        if has_attn:
            if attn_type == 'normal':
                self.attn = AttentionBlock(out_channels, attn_channels_per_head)
            elif attn_type == 'torch':
                self.attn = TorchAttentionBlock(out_channels, attn_channels_per_head)
            elif attn_type == 'flash':
                self.attn = FlashAttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x
    
    
class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, attn_channels_per_head, attn_type='normal', dropout=0.1):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout)
        if attn_type == 'normal':
            self.attn = AttentionBlock(n_channels, attn_channels_per_head)
        elif attn_type == 'torch':
            self.attn = TorchAttentionBlock(n_channels, attn_channels_per_head)
        elif attn_type == 'flash':
            self.attn = FlashAttentionBlock(n_channels, attn_channels_per_head)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x
    
class UpsampleResBlock(nn.Module):
    def __init__(self, n_channels, time_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout, up=True)

    def forward(self, x, t, z):
        return self.op(x, t, z)
    
class DownsampleResBlock(nn.Module):
    def __init__(self, n_channels, time_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout, down=True)

    def forward(self, x, t):
        return self.op(x, t)
    
    
    
    
    
class SimpleUNet(nn.Module):
    def __init__(self, 
                 img_shape:tuple = (3, 32, 32), 
                 n_channels:int = 128,
                 ch_mults:tuple = (1, 2, 2, 2),
                 is_attn:tuple = (False, True, False, False),
                 attn_channels_per_head:int = 4,
                 dropout:float = 0.1,
                 n_blocks:int = 2,
                 use_res_for_updown:bool = False,
                 t_dim:int = 512,
                 num_classes:int = 13,
                 c_dim:int = 256,
                 self_attention_type:str = 'nromal'
                 ):
        """
        * `image_shape` is the (channel, height, width) size of images.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `n_channels * ch_mults[i]`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `dropout` is the dropout rate
        * `n_blocks` is the number of ResNet blocks at each resolution
        * `t_dim` the dimension of the time embedding vector
        * `num_classes` number of classes in the dataset (used for CFG)
        * `c_dim` the dimension of the condition for CFG
        * `use_res_for_updown` indicates whether to use ResBlocks for up/down sampling (BigGAN-style)
        * `self_attention_type` is the method used for spatial self attention operation (normal, torch, flash)
        """
        super().__init__()
        
        self.img_shape = img_shape
        n_resolutions = len(ch_mults)
        
        

        self.image_proj = nn.Conv2d(img_shape[0], n_channels, kernel_size=3, padding=1)
        
        # Time embedding layer.
        # time_channels = n_channels * 4
        time_channels = t_dim
        self.time_emb = TimeEmbedding(time_channels)
        self.c_emb = nn.Embedding(num_classes, c_dim)
        
        
        # Down stages
        down = []
        in_channels = n_channels
        h_channels = [n_channels]
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks` at the same resolution
            down.append(ResAttBlock(in_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, attn_type=self_attention_type, dropout=dropout))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 1):
                down.append(ResAttBlock(out_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, attn_type=self_attention_type, dropout=dropout))
                h_channels.append(out_channels)
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                if use_res_for_updown:
                    down.append(DownsampleResBlock(out_channels, time_channels, dropout))
                else:
                    down.append(Downsample(out_channels))
                h_channels.append(out_channels)
            in_channels = out_channels
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, attn_channels_per_head, attn_type=self_attention_type, dropout=dropout)

        # Up stages
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks + 1` at the same resolution
            for _ in range(n_blocks + 1):
                up.append(ResAttBlock(in_channels + h_channels.pop(), out_channels, time_channels, is_attn[i], attn_channels_per_head, attn_type=self_attention_type, dropout=dropout))
                in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                if use_res_for_updown:
                    up.append(UpsampleResBlock(out_channels, time_channels, dropout=dropout))
                else:
                    up.append(Upsample(out_channels))
        assert not h_channels
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(out_channels, img_shape[0], kernel_size=3, padding=1)
        
        
    def forward(self, x, t, c=None, drop_mask=None, ret_activation=False):
        if not ret_activation:
            return self.forward_core(x, t, c, drop_mask)

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

        result = self.forward_core(x, t, c, drop_mask)
        for name in hooks:
            hooks[name].remove()
        return result, activation

    def forward_core(self, x, t, c=None, drop_mask=None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        * `c` condition to guide generation `[batch_size, 1]`
        * `drop_mask` has shape `[batch_size]`
        """

        t = self.time_emb(t)
        x = self.image_proj(x)
        
        # `h` will store outputs at each resolution for skip connection
        h = [x]

        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            elif isinstance(m, DownsampleResBlock):
                x = m(x, t)
            else:
                x = m(x, t).contiguous()
            h.append(x)

        x = self.middle(x, t).contiguous()

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            elif isinstance(m, UpsampleResBlock):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t).contiguous()

        return self.final(self.act(self.norm(x)))