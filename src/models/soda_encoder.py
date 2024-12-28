import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet50
from .blocks import PoseEmbedding

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
        
        if c_dim:
            self.z_dim = c_dim
        
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
                        self.c_emb = PoseEmbedding(num_freqs=c_pos_emb_freq)
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
        if c is not None:
            c = self.c_emb(c)
            c = c.permute(0, 3, 1, 2)
            x = self.rgb_linear_projector(x)
            x = torch.cat((x, c), dim=1)
        
        return self.encoder(x)
        