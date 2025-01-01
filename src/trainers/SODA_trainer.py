import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.amp import GradScaler

from ema_pytorch import EMA

# from torchmetrics import Accuracy

from torch.utils.tensorboard import SummaryWriter

import os
import socket
import datetime
from pathlib import Path
import time
from tqdm import tqdm
from functools import partial

from ..metric import LinearProbe
from ..utils import nn_utils, misc_utils

class SODATrainer():
    
    def __init__(self, 
                 max_epochs:int=400,
                 warmup_epochs:int = 20,
                 warmup_strategy:str = 'linear',
                 
                 optimizer_type:str = 'adamw',
                 enc_lr:float = 1e-4,
                 enc_dec_lr_ratio:float = 2,
                 weight_decay:float = 0.05,
                 beta1:float = 0.9,
                 beta2:float = 0.95,
                 grad_clip_norm:float = 1,
                 ema_decay:float = 0.9999,
                 
                 linear_prob_freq_e:int = 10,
                 sampling_freq_e:int = 10,
                 
                 outputs_dir:Path = Path('./outputs'),
                 write_summary:bool = True,
                 run_on_gpu:bool = True,
                 use_amp:bool = True
                 ):
        super().__init__()
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        
        outputs_dir.mkdir(exist_ok=True)
        self.outputs_dir = outputs_dir
        
        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        if self.gpu == None and run_on_gpu: raise RuntimeError("""GPU device not found!""")
        self.run_on_gpu = run_on_gpu
        self.use_amp = use_amp

        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.sampling_freq_e = sampling_freq_e if sampling_freq_e != 0 or sampling_freq_e is not None else None
        self.linear_prob_freq_e = linear_prob_freq_e if linear_prob_freq_e != 0 or linear_prob_freq_e is not None else None
        
        self.optimizer_type = optimizer_type
        self.enc_lr = enc_lr
        self.dec_lr = enc_lr / enc_dec_lr_ratio
        self.weight_decay = weight_decay
        self.optim_betas = (beta1, beta2)
        self.grad_clip_norm = grad_clip_norm
        
        if warmup_strategy == "cosine":
            self.enc_warmup_fn = lambda ep: nn_utils.cosine_warmup_lr(ep, self.enc_lr, warmup_epochs, max_epochs)
            self.dec_warmup_fn = lambda ep: nn_utils.cosine_warmup_lr(ep, self.dec_lr, warmup_epochs, max_epochs)
        elif warmup_strategy == "exponential":
            self.enc_warmup_fn = lambda ep: nn_utils.exponential_warmup_lr(ep, self.enc_lr, warmup_epochs)
            self.dec_warmup_fn = lambda ep: nn_utils.exponential_warmup_lr(ep, self.dec_lr, warmup_epochs)
        else:  # Default to linear
            self.enc_warmup_fn = lambda ep: nn_utils.linear_warmup_lr(ep, self.enc_lr, warmup_epochs)
            self.dec_warmup_fn = lambda ep: nn_utils.linear_warmup_lr(ep, self.dec_lr, warmup_epochs)
                
        self.ema_decay = ema_decay
        
        
        self.write_sum = write_summary
        if self.write_sum:
            self.writer = SummaryWriter(self.outputs_dir.joinpath('tensorboard/').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())))
            
        self.checkpoints_dir = self.outputs_dir.joinpath('checkpoints')
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.generated_samples_dir = self.outputs_dir.joinpath('generated_samples')
        self.generated_samples_dir.mkdir(exist_ok=True)
        
    def setup_data_loaders(self, dataset):
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.val_dataloader = dataset.get_val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        
        
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            views, rays, labels = batch
            return (views.to(self.gpu), rays.to(self.gpu), labels.to(self.gpu))
        else:
            return batch
        
        
    def prepare_model(self, model, enc_state_dict=None, dec_state_dict=None):
        if enc_state_dict or dec_state_dict:
            model.load_model_params(enc_state_dict, dec_state_dict)
        if self.run_on_gpu:
            model.set_device(self.gpu)
        self.model = model
        
    def prepare_EMA(self, model, ema_state_dict=None):
        ema = EMA(model, beta=self.ema_decay, update_after_step=0, update_every=1)
        if ema_state_dict:
            ema.load_state_dict(ema_state_dict)
        if self.run_on_gpu:
            ema.to(self.gpu)
        ema.eval()
        self.ema = ema
        
    
    def get_EMA(self):
        return self.ema
        
    def configure_optimizers(self, optim_state_dict=None):
        if self.optimizer_type == "adamw":
            optim = torch.optim.AdamW([
                {'params': self.model.get_encoder().parameters(), 'lr': self.enc_lr},
                {'params': self.model.get_decoder().parameters(), 'lr': self.dec_lr}
                ], betas=self.optim_betas, weight_decay=self.weight_decay)

        elif self.optimizer_type == "adam":
            optim = torch.optim.Adam([
                {'params': self.model.get_encoder().parameters(), 'lr': self.enc_lr},
                {'params': self.model.get_decoder().parameters(), 'lr': self.dec_lr}
                ], betas=self.optim_betas, weight_decay=self.weight_decay)
        else:
            raise RuntimeError("Invalide optimizer type")
        if optim_state_dict:
            optim.load_state_dict(optim_state_dict)
             
        self.optim = optim
        
        
    def setup_metrics(self, dataset):
        if self.linear_prob_freq_e:
            self.linear_probe = LinearProbe(train_set=dataset.get_train_set().get_source_dataset(),
                                            test_set=dataset.get_test_set().get_source_dataset(),
                                            num_classes=dataset.get_num_classes(),
                                            batch_size=1024,
                                            epoch=6)
        
        
        
    def fit(self, SODA, dataset, resume=False):
        self.setup_data_loaders(dataset)
        if resume:
            ckp_path = self.checkpoints_dir.joinpath('soda_ckp.pt')
            if not ckp_path.exists():
                raise RuntimeError('There is no checkpoint saved! Set the `resume` flag to False.')
            checkpoint = torch.load(ckp_path)
            self.prepare_model(SODA, checkpoint['enc_state'], checkpoint['dec_state'])
            self.configure_optimizers(checkpoint['optim_state'])
            self.prepare_EMA(SODA, checkpoint['ema_state'])
            self.epoch = checkpoint['epoch']
        else:
            self.prepare_model(SODA)
            self.configure_optimizers()
            self.prepare_EMA(SODA)
            self.epoch = 0

        self.setup_metrics(dataset)
        
        self.grad_scaler = GradScaler('cuda', enabled=self.use_amp)
        self.avg_epoch_time = misc_utils.AverageMeter()
        
        for self.epoch in range(self.epoch, self.max_epochs):
            self.fit_epoch()

        if self.write_sum:
            self.writer.flush()
            
            
    def fit_epoch(self):
        
        # ******** Training Part ********
        self.model.train()
        
        self.optim.param_groups[0]['lr'] = self.enc_warmup_fn(self.epoch)
        self.optim.param_groups[1]['lr'] = self.dec_warmup_fn(self.epoch)
        
        epoch_start_time = time.time()
        epoch_train_loss = misc_utils.AverageMeter()
        max_grad_norm = 0.0
        avg_grad_norm = misc_utils.AverageMeter()
        
        loss_ema = None
        for i, (source_batch, target_batch) in tqdm(enumerate(self.train_dataloader), total=self.num_train_batches, desc="Processing Training Epoch {}".format(self.epoch+1)):
            
            x_source, c_source, labels_s = self.prepare_batch(source_batch)
            x_target, c_target, labels_t = self.prepare_batch(target_batch)
            
            # if num_views == 1: views.squeeze(1); ray_grids.squeeze(1)
            
            self.optim.zero_grad()

            loss = self.model.training_step(x_source, x_target, c_source, c_target, self.use_amp)
            # Scale loss and backpropagate
            self.grad_scaler.scale(loss).backward()
            # Step optimizer and update scaler
            self.grad_scaler.unscale_(self.optim)
            max_grad, avg_grad = nn_utils.compute_grad_norm_stats(self.model)
            if max_grad > max_grad_norm:
                max_grad_norm = max_grad
            avg_grad_norm.update(avg_grad)
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.grad_clip_norm)
            self.grad_scaler.step(self.optim)
            self.grad_scaler.update()
            
            self.ema.update()
            if loss_ema is None: loss_ema = loss.item()
            else: loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            epoch_train_loss.update(loss.item())
        
        print(f"Max gradient norm is {max_grad_norm} and average gradient norm is {avg_grad_norm.avg}")
        print(f"Epoch{self.epoch + 1}/{self.max_epochs}, Training Loss: {epoch_train_loss.avg}, Time taken: {(time.time() - epoch_start_time)/60} minutes")
        self.avg_epoch_time.update(time.time() - epoch_start_time)
        
        if self.write_sum:
            self.writer.add_scalar('Loss/Train', epoch_train_loss.avg, self.epoch)
            self.writer.add_scalar('Loss/EMA', loss_ema, self.epoch)
            self.writer.add_scalar('LR/Encoder', self.optim.param_groups[0]['lr'], self.epoch)
            self.writer.add_scalar('LR/Decoder', self.optim.param_groups[1]['lr'], self.epoch)
            
        # ******** Saving Checkpoint ********
        if (self.epoch+1) % 50 == 0:
            print('Saving chekpoint...\n')
            path = self.checkpoints_dir.joinpath('soda_ckp.pt')
            torch.save({
                'enc_state': self.model.get_encoder().state_dict(),
                'dec_state': self.model.get_decoder().state_dict(),
                'ema_state': self.ema.state_dict(),
                'optim_state': self.optim.state_dict(),
                'epoch': self.epoch+1,
            }, path)    
            
        # ******** Validation Part ********
        if self.val_dataloader is not None and (self.epoch+1) % 10 == 0:
        
            self.model.eval()
            val_loss = misc_utils.AverageMeter()
            for i, (source_batch, target_batch) in tqdm(enumerate(self.val_dataloader), total=self.num_val_batches, desc="Processing Validation Batches"):
                x_source, c_source, labels_s = self.prepare_batch(source_batch)
                x_target, c_target, labels_t = self.prepare_batch(target_batch)
                
                with torch.no_grad():
                    loss = self.model.validation_step(x_source, x_target, c_source, c_target, self.use_amp)
                val_loss.update(loss.item())
                
            print(f"Epoch{self.epoch + 1}/{self.max_epochs}, Validation Loss: {val_loss.avg}")
            if self.write_sum:
                self.writer.add_scalar('Loss/Val', val_loss.avg, self.epoch)
                
        if self.linear_prob_freq_e and (self.epoch+1) % self.linear_prob_freq_e == 0:
            self.model.eval()
            feat_func = partial(self.ema.ema_model.encode, norm=True, use_amp=self.use_amp)
            lp_acc = self.linear_probe.evaluate(feat_func, self.model.get_encoder().get_z_dim(), device=self.gpu)
            if self.write_sum:
                self.writer.add_scalar('Metrics/Linear Probe', lp_acc, self.epoch)
            print("LinearProbe accuracy =", lp_acc)
            
        if self.sampling_freq_e and (self.epoch+1) % self.sampling_freq_e == 0:
            for s in range(4):
                self.synthesis_views(s)
            
            
            
    def synthesis_views(self, object_idx, num_source_views=1, num_target_views=24):
        self.ema.ema_model.eval()
        ema_sample_method = self.ema.ema_model.ddim_sample
        num_sources = num_source_views
        num_targets = num_target_views
        source_view, source_c, label_s = self.dataset.get_val_set().get_source_dataset().__getitem__(object_idx)
        source_view, source_c = source_view.unsqueeze(0), source_c.unsqueeze(0)
        target_views, target_cs, label_t = self.dataset.get_val_set().get_target_dataset().get_all_views(object_idx)
        
        with torch.no_grad():
            z_guide = self.ema.ema_model.encode(source_view.to(self.gpu), source_c.to(self.gpu), norm=False, use_amp=self.use_amp)
            z_guide = z_guide.repeat(num_targets, 1)
            x_gen = ema_sample_method(num_targets, z_guide, target_cs.to(self.gpu), use_amp=self.use_amp)
        # save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        x_real = (target_views.cpu() + 1) / 2
        x_gen = (x_gen.cpu() + 1) / 2
        x_all = torch.cat([x_gen, x_real])
        if self.write_sum:
            self.writer.add_images(f"Synthetized Views of Object {object_idx}", x_all, global_step=0)
        grid = torchvision.utils.make_grid(x_all, nrow=6)
        torchvision.utils.save_image(grid, self.generated_samples_dir.joinpath(f"epoch_{self.epoch+1}_image_{object_idx}_{datetime.datetime.now()}_ema.png"))
            
                
            
            
            