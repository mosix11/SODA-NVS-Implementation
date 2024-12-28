import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from torch.amp import GradScaler, autocast

import time

class LinearProbe():
    def __init__(self, train_set, test_set, num_classes, batch_size=256, lr=1e-3, epoch=1):
        self.train_set = train_set
        self.test_set = test_set
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.loss_fn = nn.CrossEntropyLoss()
        
        
        
        
    def evaluate(self, encoder, z_dim, device, use_amp=False):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True
        )
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )

        hidden_size = z_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6),
            nn.Linear(hidden_size, self.num_classes),
        )
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optim, self.epoch)
        
        grad_scaler = GradScaler('cuda', enabled=use_amp)
        # training
        self.classifier.to(device)
        self.classifier.train()
        progress_bar = tqdm(range(self.epoch), total=self.epoch, desc="Training LinearProbe on Encoder.")
        for e in progress_bar:
            for views, conds, labels in train_loader:
                # pbar.set_description("[epoch %d]: lr: %.1e" % (e, self.optim.param_groups[0]['lr']))
                # progress_bar.set_description("Training LinearProbe on Encoder,.")

                self.optim.zero_grad()
                views, conds, labels = views.to(device), conds.to(device), labels.to(device)
                with torch.no_grad():
                    feat = encoder(views, conds).detach().float()

                with autocast('cuda', enabled=use_amp):
                    logit = self.classifier(feat)
                    loss = self.loss_fn(logit, labels)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(self.optim)
                grad_scaler.update()

            self.scheduler.step()

        # testing
        all_preds = []
        all_labels = []
        self.classifier.eval()
        for i, (views, conds, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating the LinearProbe trained with Encoder's output."):
            views, conds = views.to(device), conds.to(device)
            with torch.no_grad():
                feat = encoder(views, conds).detach().float()
                logit = self.classifier(feat)
                pred = logit.argmax(dim=-1)
                all_preds.append(pred)
                all_labels.append(labels)
        pred = torch.cat(all_preds)
        label = torch.cat(all_labels)
        acc = (pred.cpu() == label).sum().item() / len(label)
        return acc