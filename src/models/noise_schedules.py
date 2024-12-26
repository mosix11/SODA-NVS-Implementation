import torch
from functools import partial



def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps # dtype = torch.float64
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def inverted_cosine_beta_schedule(timesteps, s = 0.008):
    """
    inverted cosine schedule
    as proposed in https://arxiv.org/pdf/2311.17901.pdf
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps # dtype = torch.float64
    alphas_cumprod = (2 * (1 + s) / torch.pi) * torch.arccos(torch.sqrt(t)) - s
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_schedule(beta_schedule, T, type='DDPM'):
    if beta_schedule == 'inverted':
        schedule_fn = inverted_cosine_beta_schedule
    elif beta_schedule == 'cosine':
        schedule_fn = cosine_beta_schedule
    elif beta_schedule == 'linear':
        beta1, beta2 = [1.0e-4, 0.02]
        schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)
    else:
        raise RuntimeError('Invalid beta schedule type!')
        
    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])
    elif type == 'DDIM':
        beta_t = schedule_fn(T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    # return {key: dic[key].to(device) for key in dic}
    return dic