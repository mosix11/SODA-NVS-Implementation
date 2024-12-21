import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split

import os
import sys
from pathlib import Path


class MNIST():
    
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 256,
                 img_size:tuple = (32, 32),
                 num_workers:int = 2,
                 valset_ratio:float = 0.05,
                 seed:int = 11) -> None:
        
        super().__init__()
        
        if not data_dir.exists():
            raise RuntimeError("The dat directory does not exist!")
        dataset_dir = data_dir.joinpath(Path('MNIST'))
        if not dataset_dir.exists():
            dataset_dir.mkdir()
        self.dataset_dir = dataset_dir
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.trainset_ration = 1 - valset_ratio
        self.valset_ratio = valset_ratio
        self.seed = seed
        
        transformations = [
            v2.Resize(img_size),
            v2.ToImage(),                       # Convert PIL Image/NumPy to tensor
            v2.ToDtype(torch.float32, scale=True),  # Scale to [0.0, 1.0] and set dtype
            v2.Normalize((0.1307,), (0.3081,)) # Values Specific to MNIST
        ]
        self.transformations = v2.Compose(transformations)

        self._init_loaders()


    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader

    def _init_loaders(self):
        train_dataset = datasets.MNIST(root=self.dataset_dir, train=True, transform=self.transformations, download=True)
        test_dataset = datasets.MNIST(root=self.dataset_dir, train=False, transform=self.transformations, download=True)
        
        
        if self.seed:
            generator = torch.Generator().manual_seed(self.seed)
            if self.valset_ratio == 0.0:
                trainset = train_dataset
                testset = test_dataset
            else:
                trainset, valset = random_split(train_dataset, [self.trainset_ration, self.valset_ratio], generator=generator)
                testset = test_dataset
        else:
            if self.valset_ratio == 0.0:
                trainset = train_dataset
                testset = test_dataset
            else:
                trainset, valset = random_split(train_dataset, [self.trainset_ration, self.valset_ratio])
                testset = test_dataset
            
        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = self._build_dataloader(valset) if self.valset_ratio > 0 else None
        self.test_loader = self._build_dataloader(testset)


    def _build_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader