import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchvision.transforms.v2 as transforms 
from torch.utils.data import DataLoader

from src.metric import LinearProbe
from src.datasets import NMR
from src.utils import nn_utils

from torchmetrics.image.fid import FrechetInceptionDistance

from pathlib import Path
from functools import partial
import argparse
import yaml
from tqdm import tqdm


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
    

@torch.inference_mode()
def eval_FID(model, dataset, device, use_amp=False):
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
    dataset.set_batch_size(4)
    num_targets = 24
    dataset.get_test_set().get_target_dataset().set_num_views(num_targets)
    test_loader = dataset.get_test_dataloader()
    
    def prepare_batch(batch):
        return [item.to(device) for item in batch]
    
    img_size = dataset.img_size
    fid_preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Lambda(lambda x: (x+1)/2)
    ])
    
    for i, (source_batch, target_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating the model with FID score."):
        x_source, c_source, labels_s = prepare_batch(source_batch)
        x_targets, c_targets, labels_ts = prepare_batch(target_batch)
        
        x_targets = x_targets.view(-1, 3, *img_size)
        c_targets = c_targets.view(-1, *img_size, 6)
        
        z_guide = model.encode(x_source, c_source, norm=False, use_amp=use_amp)
        
        z_guide = z_guide.repeat_interleave(num_targets, dim=0)
        
        x_gen = model.ddim_sample(z_guide.shape[0], z_guide, c_targets, use_amp=use_amp)
        
        x_real = fid_preprocess(x_targets)
        x_gen = fid_preprocess(x_gen)
        
        fid_metric.update(x_real.to('cpu'), real=True)
        fid_metric.update(x_gen.to('cpu'), real=False)
    
        fid_score = fid_metric.compute()
        print(f"FID score: {fid_score.item()}")
        
    fid_score = fid_metric.compute()
    print(f"Final FID score: {fid_score.item()}")
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="Type of evaluation to perform.", type=str, choices=['lp', 'FID'], default='lp')
    parser.add_argument('-c', '--config', help="Configuration to used for training the model.", type=str, default='NMR.yaml')
    parser.add_argument("--model", help="Which model to use for sampling between EMA and online model.", type=str, choices=['EMA', 'online'], default='EMA')
    
    args = parser.parse_args()
    
    cfg_path = Path('configs').joinpath(args.config)
    if not cfg_path.exists(): raise RuntimeError('The specified config file was not found.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)
    
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    


    
    model = None
    weights_path = weights_dir.joinpath('soda_ema.pt')
    if not weights_path.exists(): raise RuntimeError('There is no saved weights for the model. You have to train the model first!!!!')
    ema_model = torch.load(weights_path)
    if args.model == 'EMA':
        model = ema_model.ema_model 
    elif args.model == 'online':
        model = ema_model.online_model
    
    
    nmr = NMR(**cfg['dataset'])
    
    device = nn_utils.get_gpu_device()
    use_amp = False
    if device == None:
        print('No GPU found. Using CPU for evaluation!!')
        device = nn_utils.get_cpu_device()
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        use_amp = True
    
    if args.type == 'lp':
        eval_linear_probe(model, nmr, device, use_amp)
    elif args.type == 'FID':
        eval_FID(model, nmr, device, use_amp)