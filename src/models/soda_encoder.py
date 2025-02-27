import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet50
from .blocks import RayEncoder

class SodaEncoder(nn.Module):
    
    def __init__(self,
                 arch:str = 'resnet18',
                 img_shape:tuple = (3, 32, 32),
                 z_dim:int = 128,
                 c_dim:int = None,
                 c_pos_emb_freq:int = 6,
                 ) -> None :
        
        
        super().__init__()
        
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.c_dim = c_dim
        
        if arch == 'resnet18':
            net = resnet18(num_classes=z_dim,)
        elif arch == 'resnet50':
            net = resnet50(num_classes=z_dim,)
        else:
            raise RuntimeError('Invalid architecture for the Encoder!')
        
        self.encoder = []
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                self.encoder.append(nn.Flatten(1))
                self.encoder.append(module)
            else:               
                # If there is a condition grid, we concat it with the linearly projected RGB channels. 
                # If image resolution is 32 or 64 we replace first conv from 7x7 to 3x3 kernel 
                if name == 'conv1':
                    # For the first layer number of input channels are 3 and number of output channels are 64
                    self.first_channel_in = module.in_channels
                    self.frist_channel_out = module.out_channels
                    
                    if c_dim:
                        self.c_emb = RayEncoder(pos_octaves=c_pos_emb_freq, dir_octaves=c_pos_emb_freq)
                        self.rgb_linear_projector = nn.Conv2d(self.first_channel_in, self.frist_channel_out, kernel_size=1)
                        self.first_channel_in = self.frist_channel_out + c_dim
                        

                    if img_shape == (3, 32, 32) or img_shape == (3, 64, 64):
                        module = nn.Conv2d(self.first_channel_in, self.frist_channel_out,
                                            kernel_size=3, stride=1, padding=1, bias=False)
                    else:
                        module = nn.Conv2d(self.first_channel_in, self.frist_channel_out,
                                                kernel_size=7, stride=2, padding=3, bias=False)
                
                # drop first maxpooling for 32 x 32 images
                if img_shape == (3, 32, 32):  
                    if isinstance(module, nn.MaxPool2d):
                        continue
                    
                self.encoder.append(module)
                
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x, c=None):
        
        if len(x.shape) == 5: # More than one view in the batch. Shape x: [B, V, C, H, W], Shape c: [B, V, H, W, D]
            assert len(x.shape) == len(c.shape)
            xs = list(torch.unbind(x, dim=1)) # xs is a list of tensors of shape [B, C, H, W]
            cs = list(torch.unbind(c, dim=1)) # cs is a list of tensors of shape [B, H, W, D]
            zs = []
            for xi, ci in zip(xs, cs):
                pos, dirs = torch.split(ci, (3, 3), dim=-1)
                c_ = self.c_emb(pos, dirs)
                x_ = self.rgb_linear_projector(xi)
                x_ = torch.cat((x_, c_), dim=1)
                zi = self.encoder(x_)
                zs.append(zi)
            z = torch.stack(zs).mean(dim=0)
            return z
            
            
        if c is not None:
            pos, dirs = torch.split(c, (3, 3), dim=-1)
            c = self.c_emb(pos, dirs)
            x = self.rgb_linear_projector(x)
            x = torch.cat((x, c), dim=1)
        
        return self.encoder(x)
        
        
        
    def get_z_dim(self):
        return self.z_dim
    
    def get_c_dim(self):
        return self.c_dim