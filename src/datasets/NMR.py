
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as transforms 
import soft_renderer.functional as srf

from torchvision.io import read_image

import os
import gdown
import zipfile
import sys
from pathlib import Path

import tqdm

from .camera_utils import create_ray_grid

class ObjectDataset(Dataset):
    def __init__(self, obj_paths, class_ids, transform=None, num_views=2, num_channels=3) -> None:
        self.obj_paths = obj_paths
        self.transform = transform
        self.num_views = num_views
        self.class_ids = class_ids
        self.num_channels = num_channels
        
        self.elevation = 30.0
        self.distance = 2.732

        # Precompute viewpoints for all 24 views
        self.precomputed_viewpoints = srf.get_points_from_angles(
            torch.full((24,), self.distance),  # Distances
            torch.full((24,), self.elevation),  # Elevations
            -torch.arange(0, 24) * 15          # Azimuths
        )
        
        precomputed_ray_grids = []
        for viewpoint in self.precomputed_viewpoints:
            precomputed_ray_grids.append(create_ray_grid(viewpoint, H=64, W=64))

        self.precomputed_ray_grids = torch.stack(precomputed_ray_grids, dim=0)
        
        
    def __len__(self):
        return len(self.obj_paths)

    def __getitem__(self, idx):
        
        file_path = self.obj_paths[idx]
        with np.load(file_path, mmap_mode='r') as data:
            views = data['views'][:, :self.num_channels, :, :]  # Shape: (24, C, H, W)
        
        label = torch.tensor(self.class_ids.index(file_path.stem.split("_")[0])).long()
        # Randomly sample `num_views` from the 24 available views
        view_ids = np.sort(np.random.choice(24, self.num_views, replace=False))
        views_selected = views[view_ids]

        views_selected = torch.from_numpy(views_selected).float()
        ray_grids = self.precomputed_ray_grids[view_ids]
        
        

        # Apply transformations if provided
        if self.transform:
            views_selected = torch.stack([self.transform(view) for view in views_selected])

        return views_selected, ray_grids, label

    def get_all_views(self, idx):
        file_path = self.obj_paths[idx]
        with np.load(file_path, mmap_mode='r') as data:
            views = data['views'][:, :self.num_channels, :, :] # Shape: (24, C, H, W)
        
        label = torch.tensor(self.class_ids.index(file_path.stem.split("_")[0])).long()
        return torch.from_numpy(views).float(), self.precomputed_ray_grids , label

    def set_num_views(self, num_views):
        self.num_views = num_views
    
    
    
    



class NMR():
    
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 32,
                 img_size:tuple = (64, 64),
                 num_workers:int = 4,
                 num_views:int = 2,
                 load_opacity:bool = False,
                 exclude_classes:list = [],
                 seed:int = 11) -> None:
        
        super().__init__()
        
        if not data_dir.exists():
            raise RuntimeError("The data directory does not exist!")
        dataset_dir = data_dir.joinpath(Path('NMR'))
        dataset_dir.mkdir(exist_ok=True)
        self.dataset_dir = dataset_dir
        
        self._download_dataset()
        
        objects_dir = self.dataset_dir.joinpath(Path('objects'))
        objects_dir.mkdir(exist_ok=True)
        self.objects_dir= objects_dir

        
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_views = num_views
        self.num_workers = num_workers
        self.num_channels = 4 if load_opacity else 3

        self.seed = seed
        
        
        self.CLASS_IDS = [
            '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649',
            '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
        
        self.CLASS_IDS_MAP = {
            '02691156': 'Airplane',
            '02828884': 'Bench',
            '02933112': 'Cabinet',
            '02958343': 'Car',
            '03001627': 'Chair',
            '03211117': 'Display',
            '03636649': 'Lamp',
            '03691459': 'Loudspeaker',
            '04090263': 'Rifle',
            '04256520': 'Sofa',
            '04379243': 'Table',
            '04401088': 'Telephone',
            '04530566': 'Watercraft',
        }
        
        if len(exclude_classes) > 0:
            for cls in exclude_classes:
                self.CLASS_IDS.remove(cls)
                self.CLASS_IDS_MAP.pop(cls, None)

        self.elevation = 30.
        self.distance = 2.732
        

        self._preprocess_data()
        self._init_loaders()
        
        

    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    def _init_loaders(self):
        self.train_loader = self._build_dataloader(self.train_set)
        self.val_loader = self._build_dataloader(self.val_set)
        self.test_loader = self._build_dataloader(self.test_set)
    
    def _build_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader
    
        
    def _preprocess_data(self):
        self._split_views_to_files()
        
        self.train_set = ObjectDataset(self._get_split_paths('train'), self.CLASS_IDS, num_views=self.num_views, num_channels=self.num_channels)
        self.val_set = ObjectDataset(self._get_split_paths('val'), self.CLASS_IDS, num_views=self.num_views, num_channels=self.num_channels)
        self.test_set = ObjectDataset(self._get_split_paths('test'), self.CLASS_IDS, num_views=self.num_views, num_channels=self.num_channels)

    def _get_split_paths(self, split):
        paths = sorted(self.objects_dir.glob(f"*_{split}_*.npz"))  # Matches {class_id}_{split}_{obj_id}.npz
        filtered_paths = [
            path for path in paths
            if path.stem.split("_")[0] in self.CLASS_IDS
        ]
        return filtered_paths
        
    
    def _split_views_to_files(self):
        if self.objects_dir.is_dir() and not any( self.objects_dir.iterdir()): 
            mesh_dir = self.dataset_dir / Path('mesh_reconstruction')
            for split in ['train', 'val', 'test']:
                for npz_file in tqdm.tqdm(mesh_dir.glob(f'*_{split}_images.npz'), desc=f"Processing {split} set..."):
                    class_id = npz_file.stem.split('_')[0]
                    with np.load(npz_file) as data:
                        all_objects = data['arr_0']  # Shape: (num_objects, 24, C, H, W)
                        for obj_id, views in enumerate(all_objects):
                            output_path = self.objects_dir / f"{class_id}_{split}_{obj_id}.npz"
                            np.savez_compressed(output_path, views=views)

            print(f"Finished splitting objects. Output saved to {self.objects_dir}")
        else:
            print(f"Objects already processed! {self.objects_dir}")
        
    
    def _download_dataset(self):
        zip_file_path = self.dataset_dir.joinpath(Path('NMR_data.zip'))
        if zip_file_path.exists():
            print('Dataset already downloaded!')
            return
    
        try:
            google_drive_link = 'https://drive.google.com/uc?id=1fY9IWK7yEfLOmS3wUgeXM3NIivhoGhsg'
            gdown.download(google_drive_link, str(zip_file_path), quiet=False)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to download or extract the dataset: {e}")