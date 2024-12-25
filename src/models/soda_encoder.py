import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18

class SodaEncoder(nn.Module):
    
    def __init__(self, 
                 latent_dim:int ,
                 grid_condition_dim:int
                 ) -> None :
        
        
        super().__init__()
        
        net = resnet18(num_classes=latent_dim)

        self.encoder = []
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                self.encoder.append(nn.Flatten(1))
                self.encoder.append(module)
            else:
                
                # replace first conv from 7x7 to 3x3
                if name == 'conv1':
                    module = nn.Conv2d(module.in_channels, module.out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=False)
                # drop first maxpooling
                if isinstance(module, nn.MaxPool2d):
                    continue
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x)
        