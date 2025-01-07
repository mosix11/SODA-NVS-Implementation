
import torch
import torchvision
import numpy as np

from pathlib import Path
import argparse
import imageio.v2 as imageio
import yaml

from src.datasets import NMR
from src.models import SODA

from visualization import plot_multiple_views
from src.utils import nn_utils


def synthesis_views(model, sampling_method, dataset, object_idx, steps, num_source_views, num_target_views, device, use_amp):
    source_view, source_c, label_s = dataset.get_test_set().get_source_dataset().__getitem__(object_idx)
    source_view, source_c = source_view.unsqueeze(0), source_c.unsqueeze(0)
    target_views, target_cs, label_t = dataset.get_test_set().get_target_dataset().get_all_views(object_idx)
    with torch.no_grad():
        z_guide = model.encode(source_view.to(device), source_c.to(device), norm=False, use_amp=use_amp)
        z_guide = z_guide.repeat(num_target_views, 1)
        x_gen = sampling_method(num_target_views, z_guide, target_cs.to(device), steps=steps, guide_w=10, use_amp=use_amp)
    x_real = (target_views.cpu() + 1) / 2
    x_gen = (x_gen.cpu() + 1) / 2
    x_source = dataset.denormalize(source_view.cpu())
    
    
    return x_source, x_gen, x_real



def generate_GIF(tensor, path):
    tensor = (tensor.numpy() * 255).astype(np.uint8)
    imageio.mimsave(path, tensor, fps=10)
    
def torch_save_as_grid_image(tensor, path):
    grid = torchvision.utils.make_grid(tensor, nrow=6)
    torchvision.utils.save_image(grid, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="Configuration to used for training the model.", type=str, default='NMR.yaml')
    parser.add_argument("--objidx", help="Which object from the test set to use for sampling.", type=int, default=-1)
    parser.add_argument("--model", help="Which model to use for sampling between EMA and online model.", type=str, choices=['EMA', 'online'], default='EMA')
    parser.add_argument("--method", help="Which sampling method to use between DDPM and DDIM.", type=str, choices=['DDIM', 'DDPM'], default='DDIM')
    parser.add_argument("--nfe", help="NFE or number of feature calls during sampling.", type=int, default=200)
    parser.add_argument('--sources', help="Number of source views to use for generating latent code", type=int, default=1)
    
    args = parser.parse_args()
    
    cfg_path = Path('configs').joinpath(args.config)
    if not cfg_path.exists(): raise RuntimeError('The specified config file was not found.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)
    
    weights_dir = Path('weights')
    model = None
    weights_path = weights_dir.joinpath('soda_ema.pt')
    if not weights_path.exists(): raise RuntimeError('There is no saved weights for the model. You have to train the model first!!!!')
    ema_model = torch.load(weights_path)
    if args.model == 'EMA':
        model = ema_model.ema_model 
    elif args.model == 'online':
        model = ema_model.online_model
        
    if args.method == 'DDIM':
        sampling_method = model.ddim_sample
    elif args.method == 'DDPM':
        sampling_method = model.ddpm_sample 
        
    NFE = args.nfe
    
    model.eval()
    nmr = NMR(**cfg['dataset'])
    
    saving_dir = Path('outputs/sampled_GIFs')
    saving_dir.mkdir(exist_ok=True)
    
    device = nn_utils.get_gpu_device()
    use_amp = False
    if device == None:
        print('No GPU found. Sampling on CPU!!')
        device = nn_utils.get_cpu_device()
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        use_amp = True
    
    num_source_views = args.sources
    num_target_views = 24
    
    if args.objidx == -1: object_index = np.random.randint(0, nmr.get_test_set().get_source_dataset().__len__())
    else: object_index = args.objidx
    
    x_source, x_gen, x_real = synthesis_views(model, sampling_method, nmr, object_index, NFE/2, num_source_views, num_target_views, device, use_amp)
    
    # x_gen.permute(0, 2, 3, 1), x_real.permute(0, 2, 3, 1)
    # torch.cat([x_gen, x_real])
    # Path('outputs/sampled_GIFs').joinpath(f"{object_index}-images.png")
    torchvision.utils.save_image(x_source, saving_dir.joinpath(f"{object_index}-source.png"))
    torch_save_as_grid_image(x_real, saving_dir.joinpath(f"{object_index}-real.png"))
    torch_save_as_grid_image(x_gen, saving_dir.joinpath(f"{object_index}-gen.png"))
    generate_GIF(x_gen.permute(0, 2, 3, 1), saving_dir.joinpath(f"{object_index}-gen.gif"))
    generate_GIF(x_real.permute(0, 2, 3, 1), saving_dir.joinpath(f"{object_index}-real.gif"))
    print(f"The sampled views was saved at {saving_dir}/{object_index}-*.*")
    
