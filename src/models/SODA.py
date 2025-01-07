import torch
import torch.nn as nn

from tqdm import tqdm

from torch.amp import autocast

from .noise_schedules import get_schedule
from .unet import UNet
from .soda_encoder import SodaEncoder

class SODA(nn.Module):
    def __init__(self, 
                 encoder:SodaEncoder,
                 decoder:UNet,
                 beta_schedule:str = 'inverted',
                 T:int = 1000,
                 z_drop_prob:float = 0.12,
                #  c_drop_prob:float = 0.1,
                 )->None:
        ''' SODA proposed by "SODA: Bottleneck Diffusion Models for Representation Learning", and \
            DDPM proposed by "Denoising Diffusion Probabilistic Models", as well as \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                encoder: A network (e.g. ResNet) which performs image->latent mapping.
                decoder: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                beta_schedule, T, z_drop_prob
        '''
        super(SODA, self).__init__()
        

        self.encoder = encoder
            
        self.decoder = decoder
        
        self.enc_img_shape = encoder.img_shape
        self.dec_img_shape = decoder.img_shape
        
        self.ddpm_sche = get_schedule(beta_schedule, T, 'DDPM')
        self.ddim_sche = get_schedule(beta_schedule, T, 'DDIM')
        
        self.T = T
        self.z_drop_prob = z_drop_prob
        # self.c_drop_prob = c_drop_prob
        
        self.loss = nn.MSELoss()
        
        self.device = torch.device('cpu')
        
        
    def set_device(self, device):
        self.device = device
        self.ddpm_sche = {key: self.ddpm_sche[key].to(device) for key in self.ddpm_sche}
        self.ddim_sche = {key: self.ddim_sche[key].to(device) for key in self.ddim_sche}
        self.encoder.to(device)
        self.decoder.to(device)
        
    def load_model_params(self, enc_state_dict=None, dec_state_dict=None):
        if enc_state_dict: self.encoder.load_state_dict(enc_state_dict)
        if dec_state_dict: self.decoder.load_state_dict(dec_state_dict)
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(1, self.T + 1, (x.shape[0], )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise
        
        
    def training_step(self, x_source, x_target, c_source=None, c_target=None, use_amp=False):
        pred_noise, noise = self(x_source, x_target, c_source, c_target, use_amp)
        with autocast('cuda', enabled=use_amp):
            loss = self.loss(noise, pred_noise)
        return loss
    
    def validation_step(self, x_source, x_target, c_source=None, c_target=None, use_amp=False):
        pred_noise, noise = self(x_source, x_target, c_source, c_target, use_amp)
        with autocast('cuda', enabled=use_amp):
            loss = self.loss(noise, pred_noise)
        return loss
        
    def forward(self, x_source, x_target, c_source=None, c_target=None, use_amp=False):
        ''' Training with simple noise prediction loss.

            Args:
                x_source: Encoder input image tensor.
                x_target: Denoiser input image tensor.
                c_source: conditioning tensor for the Encoder input.
                c_target: conditioning tensor for the Denoiser input.
            Returns:
                The simple MSE loss.
        '''
        
        x_noised, t, noise = self.perturb(x_target, t=None)

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros(x_noised.shape[0]) + self.z_drop_prob).to(self.device)
        with autocast('cuda', enabled=use_amp):
            z = self.encoder(x_source, c_source)
            pred_noise = self.decoder(x_noised, t / self.T, z, mask, c_target)
        return pred_noise, noise

    def encode(self, x, c=None, norm=False, use_amp=False):
        with autocast('cuda', enabled=use_amp):
            z = self.encoder(x, c)
        if norm:
            z = torch.nn.functional.normalize(z)
        return z
    
    
    def ddim_sample(self, n_sample, z_guide, c_cond, steps=100, eta=0.0, guide_w=2, notqdm=False, use_amp=False):
        ''' Sampling with DDIM sampler. Actual NFE is `2 * steps`.

            Args:
                n_sample: The batch size.
                z_guide: The latent code extracted from real images (for guidance).
                c_cond: Conditioning tensor for conditional generation.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[-1, 1]`.
        '''
        size = self.dec_img_shape
        sche = self.ddim_sche
        z_guide, mask = self.prepare_condition_(n_sample, z_guide)
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.T, self.T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm, desc="Sampling using DDIM"):
            t_is = torch.tensor([time / self.T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, z_guide, mask, c_cond, guide_w, alpha, use_amp)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        return x_i
    
    def ddpm_sample(self, n_sample, z_guide, c_cond, steps=100, guide_w=2, notqdm=False, use_amp=False):
        ''' Sampling with DDPM sampler. Actual NFE is `steps`.

            Args:
                n_sample: The batch size.
                z_guide: The latent code extracted from real images (for guidance).
                c_cond: Conditioning tensor for conditional generation.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[-1, 1]`.
        '''
        size = self.dec_img_shape
        sche = self.ddpm_sche  # Reuse the same schedule dictionary.
        z_guide, mask = self.prepare_condition_(n_sample, z_guide)
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.T, self.T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm, desc="Sampling using DDPM"):
            t_is = torch.tensor([time / self.T]).to(self.device).repeat(n_sample)

            alpha = sche["alphabar_t"][time]
            alpha_next = sche["alphabar_t"][time_next]

            # Predict epsilon and denoised image x0
            eps, x0_t = self.pred_eps_(x_i, t_is, z_guide, mask, c_cond, guide_w, alpha, use_amp)

            # Compute the mean for the next step
            mean = alpha_next.sqrt() * x0_t + (1 - alpha_next).sqrt() * eps

            # Add stochastic noise if not at the final step
            if time_next > 0:
                std_dev = ((1 - alpha_next) / alpha_next).sqrt()
                noise = torch.randn_like(x_i).to(self.device)
                x_i = mean + std_dev * noise
            else:
                x_i = mean

        return x_i
    
    def pred_eps_(self, x, t, z, mask, c, guide_w, alpha, use_amp, clip_x=False):
        def pred_cfg_eps_double_batch():
            # double batch
            x_double = x.repeat(2, 1, 1, 1)
            c_double = c.repeat(2, 1, 1, 1)
            t_double = t.repeat(2)
            # print(x_double.shape, c_double.shape, t_double.shape, z.shape, mask.shape)
            with autocast('cuda', enabled=use_amp):
                eps = self.decoder(x_double, t_double, z, mask, c_double).float()
            n_sample = eps.shape[0] // 2
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            assert eps1.shape == eps2.shape
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            return eps

        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
        eps = pred_cfg_eps_double_batch()
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised

    def prepare_condition_(self, n_sample, z_guide):
        z_guide = z_guide.repeat(2, 1)

        # 0 for conditional, 1 for unconditional
        mask = torch.zeros(z_guide.shape[0]).to(self.device)
        mask[n_sample:] = 1.
        return z_guide, mask