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
    
    batch_size = 32
    image_size = (32, 32)
    num_views = 2
    nmr = NMR(
        batch_size=batch_size,
        img_size=image_size,
        num_views=num_views,
        exclude_classes=[]
    )
    
    soda = SODA(
        use_encoder=True,
        encoder_arch='resnet18',
        enc_img_shape=(3, *image_size),
        z_dim=128,
        c_pos_emb_freq=6,
        c_dim=6*2*6,
        dec_img_shape=(3, *image_size),
    )
    
    trainer = SODATrainer(
        max_epochs=400,
        warmup_epochs=20,
        )
    
    trainer.fit(soda, nmr, resume=False)