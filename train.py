import torch
import os
import sys
import argparse
import yaml
from pathlib import Path

from src.datasets import NMR
from src.models import SODA, SodaEncoder, UNet
from src.trainers import SODATrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="Configuration to use for training the model.", type=str, default='NMR.yaml')
    args = parser.parse_args()
    
    cfg_path = Path('configs').joinpath(args.config)
    if not cfg_path.exists(): raise RuntimeError('The specified config file was not found.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)
    
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    

    nmr = NMR(**cfg['dataset'])
    # print(nmr.get_train_set().__len__())
    # print(nmr.get_val_set().__len__())
    # print(nmr.get_test_set().__len__())
    encoder = SodaEncoder(**cfg['encoder'])
    denoiser = UNet(**cfg['denoiser'])
    soda = SODA(
        encoder=encoder,
        decoder=denoiser,
        **cfg['SODA']
    )
    trainer = SODATrainer(**cfg['trainer'])
    trainer.fit(soda, nmr, resume=False)
    torch.save(soda, weights_dir.joinpath('soda.pt'))
    torch.save(trainer.get_EMA(), weights_dir.joinpath('soda_ema.pt'))