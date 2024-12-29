import torch
import os
import sys
import argparse
from pathlib import Path

from src.datasets import NMR
from src.models import SODA
from src.trainers import SODATrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    batch_size = 64
    image_size = (32, 32)
    num_views = 2
    nmr = NMR(
        batch_size=batch_size,
        img_size=image_size,
        num_views=num_views,
        exclude_classes=[],
        num_workers=12
    )
    
    soda = SODA(
        use_encoder=True,
        encoder_arch='resnet18',
        enc_img_shape=(3, *image_size),
        z_dim=128,
        c_pos_emb_freq=24,
        c_dim=6*2*24,
        dec_img_shape=(3, *image_size),
    )
    
    trainer = SODATrainer(
        max_epochs=800,
        warmup_epochs=20,
        
        linear_prob_freq_e = 40,
        sampling_freq_e = 20,
        write_summary = True,
        run_on_gpu = True,
        use_amp = True
    )
    
    trainer.fit(soda, nmr, resume=False)
    torch.save(soda, weights_dir.joinpath('soda.pt'))