import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchvision.transforms.v2 as transforms 
from torch.utils.data import DataLoader

from src.models import SodaEncoder, UNet, SODA
from src.metric import LinearProbe
from src.datasets import NMR
from src.utils import nn_utils

from ema_pytorch import EMA

from torchmetrics.image import FrechetInceptionDistance, StructuralSimilarityIndexMeasure
from pytorch_fid import fid_score

from pathlib import Path
from functools import partial
import argparse
import yaml
from tqdm import tqdm

import subprocess

def eval_linear_probe(model, dataset, device, use_amp=False):
    linear_probe = LinearProbe(train_set=dataset.get_train_set().get_source_dataset(),
                                        test_set=dataset.get_test_set().get_source_dataset(),
                                        num_classes=dataset.get_num_classes(),
                                        batch_size=512,
                                        epoch=6)
    model.eval()
    feat_func = partial(model.encode, norm=True, use_amp=use_amp)
    lp_acc = linear_probe.evaluate(feat_func, model.get_encoder().get_z_dim(), device=device)
    print("LinearProbe accuracy =", lp_acc)
    

# @torch.inference_mode()
# def eval_FID(model, dataset, device, use_amp=False):
#     model.eval()
#     fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
#     dataset.set_batch_size(4)
#     num_targets = 24
#     dataset.get_test_set().get_target_dataset().set_num_views(num_targets)
#     test_loader = dataset.get_test_dataloader()
    
#     def prepare_batch(batch):
#         return [item.to(device) for item in batch]
    
#     img_size = dataset.img_size
#     fid_preprocess = transforms.Compose([
#         transforms.Resize((299, 299)),
#         transforms.Lambda(lambda x: (x+1)/2)
#     ])
    
#     for i, (source_batch, target_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating the model with FID score."):
#         x_source, c_source, labels_s = prepare_batch(source_batch)
#         x_targets, c_targets, labels_ts = prepare_batch(target_batch)
        
#         x_targets = x_targets.view(-1, 3, *img_size)
#         c_targets = c_targets.view(-1, *img_size, 6)
        
#         z_guide = model.encode(x_source, c_source, norm=False, use_amp=use_amp)
        
#         z_guide = z_guide.repeat_interleave(num_targets, dim=0)
        
#         x_gen = model.ddim_sample(z_guide.shape[0], z_guide, c_targets, use_amp=use_amp)
        
#         x_real = fid_preprocess(x_targets)
#         x_gen = fid_preprocess(x_gen)
        
#         fid_metric.update(x_real.to('cpu'), real=True)
#         fid_metric.update(x_gen.to('cpu'), real=False)

    
#         fid_score = fid_metric.compute()
#         print(f"FID score: {fid_score.item()}")
        
#     fid_score = fid_metric.compute()
#     print(f"Final FID score: {fid_score.item()}")

@torch.inference_mode()
def eval_FID(model, dataset, device, use_amp=False):
    model.eval()
    store_dir = Path('outputs/fid_temp_dir')
    store_dir.mkdir(exist_ok=True)
    real_imgs_dir = store_dir / Path('real')
    real_imgs_dir.mkdir(exist_ok=True)
    gen_imgs_dir = store_dir / Path('gen')
    gen_imgs_dir.mkdir(exist_ok=True)
    
    batch_size = 4
    dataset.set_batch_size(batch_size)
    num_targets = 24
    dataset.get_test_set().get_target_dataset().set_num_views(num_targets)
    test_loader = dataset.get_test_dataloader()
    
    def prepare_batch(batch):
        return [item.to(device) for item in batch]
    
    img_size = dataset.img_size
    fid_preprocess = transforms.Compose([
        # transforms.Resize((299, 299)),
        transforms.Lambda(lambda x: (x+1)/2),
    ])
    to_pil = transforms.ToPILImage()
    
    num_batches = 10
    for i, (source_batch, target_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating the model with FID score."):
        x_source, c_source, labels_s = prepare_batch(source_batch)
        x_targets, c_targets, labels_ts = prepare_batch(target_batch)
        
        x_targets = x_targets.view(-1, 3, *img_size)
        c_targets = c_targets.view(-1, *img_size, 6)
        
        z_guide = model.encode(x_source, c_source, norm=False, use_amp=use_amp)
        
        z_guide = z_guide.repeat_interleave(num_targets, dim=0)
        
        x_gen = model.ddim_sample(z_guide.shape[0], z_guide, c_targets, use_amp=use_amp)
        

        x_real = fid_preprocess(x_targets).to('cpu')
        x_gen = fid_preprocess(x_gen).to('cpu')
        
        for j, (img_real, img_gen) in enumerate(zip(x_real, x_gen)):
            img_name = f"{(i*batch_size*24+j):05}.png"
            real_img_path = real_imgs_dir / Path(img_name)
            gen_img_path = gen_imgs_dir / Path(img_name)
            img_real, img_gen = to_pil(img_real), to_pil(img_gen)
            img_real.save(real_img_path)
            img_gen.save(gen_img_path)
        
        if i + 1 >= num_batches:
            break
        
    cli_command = [
        "CUDA_VISIBLE_DEVICES=0", 
        "pytorch-fid", 
        "outputs/fid_temp_dir/real", 
        "outputs/fid_temp_dir/gen"
    ]
    result = subprocess.run(
        " ".join(cli_command),  
        shell=True,        
        capture_output=True, 
        text=True
    )
    print(result.stdout)
    # print("Standard Error:", result.stderr)
    print("Exit Code:", result.returncode)

@torch.inference_mode()
def eval_SSIM(model, dataset, device, use_amp=False):
    model.eval()
    ssim = StructuralSimilarityIndexMeasure(gaussian_kernel=False, data_range=1.0)
    
    dataset.set_batch_size(4)
    num_targets = 24
    dataset.get_test_set().get_target_dataset().set_num_views(num_targets)
    test_loader = dataset.get_test_dataloader()
    
    img_size = dataset.img_size
    
    def prepare_batch(batch):
        return [item.to(device) for item in batch]
    
    ssim_preprocess = transforms.Compose([
        transforms.Lambda(lambda x: (x+1)/2)
    ])
    
    for i, (source_batch, target_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating the model with SSIM score."):
        x_source, c_source, labels_s = prepare_batch(source_batch)
        x_targets, c_targets, labels_ts = prepare_batch(target_batch)
        
        x_targets = x_targets.view(-1, 3, *img_size)
        c_targets = c_targets.view(-1, *img_size, 6)
        
        z_guide = model.encode(x_source, c_source, norm=False, use_amp=use_amp)
        z_guide = z_guide.repeat_interleave(num_targets, dim=0)
        x_gen = model.ddim_sample(z_guide.shape[0], z_guide, c_targets, use_amp=use_amp)
        
        x_real = ssim_preprocess(x_targets).to('cpu')
        x_gen = ssim_preprocess(x_gen).to('cpu')
        
        print(ssim(x_gen, x_real))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="Type of evaluation to perform.", type=str, choices=['lp', 'FID', 'SSIM'], default='lp')
    parser.add_argument('-c', '--config', help="Configuration to used for training the model.", type=str, default='NMR.yaml')
    parser.add_argument("--model", help="Which model to use for sampling between EMA and online model.", type=str, choices=['EMA', 'online'], default='EMA')
    
    args = parser.parse_args()
    
    cfg_path = Path('configs').joinpath(args.config)
    if not cfg_path.exists(): raise RuntimeError('The specified config file was not found.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)
    
    weights_dir = Path('weights')
    
    model = None
    weights_path = weights_dir.joinpath('soda_ema.pt')
    if not weights_path.exists(): raise RuntimeError('There is no saved weights for the model. You have to train the model first!!!!')
    
    
    encoder = SodaEncoder(**cfg['encoder'])
    denoiser = UNet(**cfg['denoiser'])
    soda = SODA(
        encoder=encoder,
        decoder=denoiser,
        **cfg['SODA']
    )
    ema_model = EMA(soda, beta=cfg['trainer']['ema_decay'], update_after_step=0, update_every=1)
    weights = torch.load(weights_path)
    ema_model.load_state_dict(weights)
    if args.model == 'EMA':
        model = ema_model.ema_model 
    elif args.model == 'online':
        model = ema_model.online_model
    
    
    nmr = NMR(**cfg['dataset'])
    
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
        
    model.set_device(device)
    
    if args.type == 'lp':
        eval_linear_probe(model, nmr, device, use_amp)
    elif args.type == 'FID':
        eval_FID(model, nmr, device, use_amp)
        
    elif args.type == 'SSIM':
        eval_SSIM(model, nmr, device, use_amp)