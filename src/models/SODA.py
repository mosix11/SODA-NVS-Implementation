import torch
import torch.nn as nn

from .noise_schedules import get_schedule
from .unet import UNet
from .soda_encoder import SodaEncoder

class SODA(nn.Module):
    def __init__(self, 
                 encoder_arch:str = 'resnet18',
                 enc_img_shape:tuple = (3, 32, 32),
                 z_dim:int = 128,
                 c_dim:int = None,
                 c_pos_emb_freq:int = 10,
                 
                 dec_img_shape:tuple = (3, 32, 32),
                 dec_dropout:float = 0.1,
                 self_att_type:str = 'normal',
                 
                 beta_schedule:str = 'inverted',
                 T:int = 1000,
                 t_dim:int = 512,
                 condition_drop_prob:float = 0.1
                 )->None:
        ''' SODA proposed by "SODA: Bottleneck Diffusion Models for Representation Learning", and \
            DDPM proposed by "Denoising Diffusion Probabilistic Models", as well as \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                encoder: A network (e.g. ResNet) which performs image->latent mapping.
                decoder: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                beta_schedule, n_T, drop_prob
        '''
        super(SODA, self).__init__()
        
        self.encoder = SodaEncoder(arch=encoder_arch,
                                   img_shape=enc_img_shape,
                                   z_dim=z_dim,
                                   c_dim=c_dim,
                                   c_pos_emb_freq=c_pos_emb_freq)
        
        self.decoder = UNet(img_shape=dec_img_shape,
                            n_channels=128,
                            ch_mults=(1, 2, 2, 2),
                            is_attn=(False, True, False, False),
                            attn_channels_per_head=None,
                            dropout=dec_dropout,
                            n_blocks=2,
                            use_res_for_updown=False,
                            t_dim=t_dim,
                            z_dim=z_dim,
                            c_dim=c_dim,
                            self_attention_type=self_att_type
                            )
        
        self.ddpm_sche = get_schedule(beta_schedule, T, 'DDPM')
        self.ddim_sche = get_schedule(beta_schedule, T, 'DDIM')
        
        self.T = T
        self.condition_drop_prob = condition_drop_prob
        self.loss = nn.MSELoss()
        
        self.device = torch.device('cpu')
        
        
    def set_device(self, device):
        self.device = device
        self.ddpm_sche.to(device)
        self.ddim_sche.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        
    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise
        
        
    def forward(self, x_source, x_target, c=None):
        ''' Training with simple noise prediction loss.

            Args:
                x_source: Encoder input image tensor.
                x_target: Denoiser input image tensor.
                c: conditioning tensor.
            Returns:
                The simple MSE loss.
        '''
        
        x_noised, t, noise = self.perturb(x_target, t=None)

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros(x_noised.shape[0]) + self.drop_prob).to(self.device)

        z = self.encoder(x_source, c)
        x_recon = self.decoder(x_noised, t / self.T, z, mask, c)
        return self.loss(noise, )

    def encode(self, x, norm=False, use_amp=False):
        with autocast(enabled=use_amp):
            z = self.encoder(x)
        if norm:
            z = torch.nn.functional.normalize(z)
        return z