
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as transforms 
import soft_renderer.functional as srf

from torchvision.io import decode_image

import os
import gdown
import zipfile
import requests
import sys
from pathlib import Path
import shutil

from functools import partial

import tqdm
from ..utils import misc_utils

from .camera_utils import compute_ray_grids, compute_ray_grids_for_views, check_ray_grid_with_pointcloud


class SourceTargetDataset(Dataset):
    def __init__(self, source_ds, target_ds):
        self.source_ds = source_ds
        self.target_ds = target_ds

    def __getitem__(self, index):
        views_s, cond_s, labels_s = self.source_ds.__getitem__(index)
        views_t, cond_t, labels_t = self.target_ds.__getitem__(index)
        assert labels_s == labels_t
        return (views_s, cond_s, labels_s), (views_t, cond_t, labels_t)

    def __len__(self):
        l1 = self.source_ds.__len__()
        l2 = self.target_ds.__len__()
        assert l1 == l2
        return l1
    
    def get_source_dataset(self):
        return self.source_ds
    
    def get_target_dataset(self):
        return self.target_ds

class ObjectDataset(Dataset):
    def __init__(self, obj_paths, class_ids, img_size = (32, 32), transform=None, num_views=1) -> None:
        self.obj_paths = obj_paths
        self.img_size = img_size
        self.transform = transform
        self.num_views = num_views
        self.class_ids = class_ids
        
        
    def __len__(self):
        return len(self.obj_paths)

    def __getitem__(self, idx):
        
        obj_path = self.obj_paths[idx]
        
        parts = obj_path.parts
        class_id = parts[-2]
        label = torch.tensor(self.class_ids.index(class_id)).long()
        
        view_idxs = np.sort(np.random.choice(24, self.num_views, replace=False))
        view_files_names = [f"00{id:02d}.png" for id in view_idxs]
        
        
        views = [decode_image(obj_path.joinpath('image').joinpath(vfn)) for vfn in view_files_names]

        # Apply transformations if provided
        if self.transform:
            views = torch.stack([self.transform(view) for view in views], dim=0)
            
            
        ray_grids = compute_ray_grids_for_views(obj_path.joinpath('cameras.npz'), H=self.img_size[0],
                                                W=self.img_size[1], views_idx=view_idxs, use_canonical=True)
        # ray_grids = compute_ray_grids(obj_path.joinpath('cameras.npz'), H=self.img_size[0], W=self.img_size[1], use_canonical=True)
        
        # check_ray_grid_with_pointcloud(obj_path.joinpath('cameras.npz'), obj_path.joinpath('pointcloud.npz'), ray_grids, view_idx=0,
        #                                         H=self.img_size[0],
        #                                         W=self.img_size[1])

        if self.num_views == 1:
            views = views.squeeze(0)
            ray_grids = ray_grids.squeeze(0)
        return views, ray_grids, label

    def get_all_views(self, idx):
        obj_path = self.obj_paths[idx]
        
        parts = obj_path.parts
        class_id = parts[-2]
        label = torch.tensor(self.class_ids.index(class_id)).long()
        
        view_ids = np.arange(0, 24)
        view_files_names = [f"00{id:02d}.png" for id in view_ids]
        views = [decode_image(obj_path.joinpath('image').joinpath(vfn)) for vfn in view_files_names]
        
        # Apply transformations if provided
        if self.transform:
            views = torch.stack([self.transform(view) for view in views], dim=0)
            
        ray_grids = compute_ray_grids(obj_path.joinpath('cameras.npz'), H=self.img_size[0], W=self.img_size[1], use_canonical=True)
        
        return views, ray_grids, label
    
    

    def set_num_views(self, num_views):
        self.num_views = num_views
    
    
    
    
class NMR():
    
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 32,
                 img_size:tuple = (32, 32),
                 num_workers:int = 8,
                 num_views:int = 2,
                 exclude_classes:list = [],
                 seed:int = 11) -> None:
        
        super().__init__()
        
        if not data_dir.exists():
            raise RuntimeError("The data directory does not exist!")
        dataset_base_dir = data_dir.joinpath(Path('NMR'))
        dataset_base_dir.mkdir(exist_ok=True)
        self.dataset_base_dir = dataset_base_dir
        
        objects_base_dir = self.dataset_base_dir.joinpath(Path('NMR_Dataset'))
        self.objects_base_dir= objects_base_dir
        
        self._download_dataset()
        
        

        
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_views = num_views
        self.num_workers = num_workers

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

        self.num_classes = len(self.CLASS_IDS)
        
        # self.dataset_mean = (0.09258475, 0.09884189, 0.10409881, 0.17108302) # RGBA
        # self.dataset_std = (0.21175679, 0.22688305, 0.24043436, 0.36608162) # RGBA
        self.dataset_mean = (0.90754463, 0.90124703, 0.89597464) # RGB
        self.dataset_std = (0.21146126, 0.22668979, 0.24028395) # RGB
        
        encoder_transformations = [
            transforms.Resize(img_size),
            transforms.ToImage(),                       # Convert PIL Image/NumPy to tensor
            transforms.ToDtype(torch.float32, scale=True),  # Scale to [0.0, 1.0] and set dtype
            transforms.Normalize(self.dataset_mean, self.dataset_std) # Values Specific to NMR calculated from all splits
        ]
        self.encoder_transformations = transforms.Compose(encoder_transformations)
        
        denoiser_transformations = [
            transforms.Resize(img_size),
            transforms.ToImage(),                       # Convert PIL Image/NumPy to tensor
            transforms.ToDtype(torch.float32, scale=True),  # Scale to [0.0, 1.0] and set dtype
            transforms.Lambda(lambda x: x * 2 - 1),         # Scale [0, 1] to [-1, 1]
        ]
        self.denoizer_transformation = transforms.Compose(denoiser_transformations)
        

        self._build_datasets()
        self._init_loaders()
        
    
    def get_dataset_stats(self):
        return self.dataset_mean, self.dataset_std    

    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    def get_train_set(self):
        return self.train_set
    
    def get_val_set(self):
        return self.val_set
    
    def get_test_set(self):
        return self.test_set
    
    def get_num_classes(self):
        return self.num_classes
    
    def denormalize(self, batch_tensor):
        """
        Denormalize a batch of tensor images channel-wise using the provided mean and std.
        Assumes the input is a normalized tensor with shape [B, C, H, W].
        The mean and std should be lists or tensors of length equal to the number of channels.
        """
        mean = torch.tensor(self.dataset_mean).view(1, -1, 1, 1)  # Reshape to [1, C, 1, 1] for broadcasting
        std = torch.tensor(self.dataset_std).view(1, -1, 1, 1)    # Reshape to [1, C, 1, 1] for broadcasting
        batch_tensor = batch_tensor * std + mean     # Apply channel-wise denormalization
        return torch.clip(batch_tensor, 0, 1)        # Clip values to [0, 1]
    
    
    
    def get_label_tag(self, label_idx):
        return self.CLASS_IDS_MAP[self.CLASS_IDS[label_idx]]
    
    def _init_loaders(self):
        self.train_loader = self._build_dataloader(self.train_set)
        self.val_loader = self._build_dataloader(self.val_set)
        self.test_loader = self._build_dataloader(self.test_set)
    
    def _build_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader
    
        
    def _build_datasets(self):
        train_split_paths = self._get_split_paths('train')
        val_split_paths = self._get_split_paths('val')
        test_split_paths = self._get_split_paths('test')
        
        ObjectDatasetPartial = partial(ObjectDataset, class_ids=self.CLASS_IDS, img_size=self.img_size)
        num_source_views = self.num_views -1
        num_target_views = 1
        
        train_source_ds = ObjectDatasetPartial(obj_paths=train_split_paths, transform=self.encoder_transformations, num_views=num_source_views)
        train_target_ds = ObjectDatasetPartial(obj_paths=train_split_paths, transform=self.denoizer_transformation, num_views=num_target_views)
        
        val_source_ds = ObjectDatasetPartial(obj_paths=val_split_paths, transform=self.encoder_transformations, num_views=num_source_views)
        val_target_ds = ObjectDatasetPartial(obj_paths=val_split_paths, transform=self.denoizer_transformation, num_views=num_target_views)
        
        test_source_ds = ObjectDatasetPartial(obj_paths=test_split_paths, transform=self.encoder_transformations, num_views=num_source_views)
        test_target_ds = ObjectDatasetPartial(obj_paths=test_split_paths, transform=self.denoizer_transformation, num_views=num_target_views)
        
        self.train_set = SourceTargetDataset(train_source_ds, train_target_ds)
        self.val_set = SourceTargetDataset(val_source_ds, val_target_ds)
        self.test_set = SourceTargetDataset(test_source_ds, test_target_ds)


    def _get_split_paths(self, split):
        all_objects_paths = []
        for class_id in self.CLASS_IDS:
            base_class_path = self.objects_base_dir.joinpath(Path(class_id))
            with open(os.path.join(base_class_path, f'softras_{split}.lst')) as f:
                objects_ids = f.readlines()
            objects_ids = [obj_id.rstrip() for obj_id in objects_ids if len(obj_id) > 1]
            objects_paths = [base_class_path.joinpath(object_id) for object_id in objects_ids]
            all_objects_paths.extend(objects_paths)
            
        return all_objects_paths
        
        
    def _download_dataset(self):
        zip_file_path = self.dataset_base_dir.joinpath(Path('NMR_Dataset.zip'))
        if zip_file_path.exists():
            print('Dataset already downloaded!')
        else:
            try:
                url = "https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip"
                print("Starting download...")
                misc_utils.download_file_fast(url, zip_file_path.absolute())
                
            except Exception as e:
                raise RuntimeError(f"Failed to download the dataset: {e}")
        if self.objects_base_dir.exists():
            print("Dataset already extracted!")
        else:
            try:
                print("Extracting the file...")
                self.objects_base_dir.mkdir(exist_ok=False)
                misc_utils.extract_zip_multithreaded(zip_file_path.absolute(), self.dataset_base_dir.absolute())
                print(f"Extraction completed. Files are available in: {self.objects_base_dir}")
            except Exception as e:
                shutil.rmtree(self.objects_base_dir)
                raise RuntimeError(f"Failed to extraxt the dataset: {e}")
    
    
    