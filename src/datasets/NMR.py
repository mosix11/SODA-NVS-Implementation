

import os
import gdown
import zipfile
import sys
from pathlib import Path

import torch
import numpy as np
import tqdm


class NMR():
    
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 256,
                 img_size:tuple = (64, 64),
                 num_workers:int = 2,
                 seed:int = 11) -> None:
        
        super().__init__()
        
        if not data_dir.exists():
            raise RuntimeError("The data directory does not exist!")
        dataset_dir = data_dir.joinpath(Path('NMR'))
        if not dataset_dir.exists():
            dataset_dir.mkdir()
        self.dataset_dir = dataset_dir
        
        self._download_dataset()

        
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        self.seed = seed
        
        self.CLASS_IDS_ALL = (
            '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
            '03691459,04090263,04256520,04379243,04401088,04530566')
        
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

        self.elevation = 30.
        self.distance = 2.732
        
        self._load_data()
        
        
        
        
    def _load_data(self):
        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(list(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            voxels.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels
    
    
    
    def _download_dataset(self):
        zip_file_path = self.dataset_dir.joinpath(Path('NMR_data.zip'))
        if zip_file_path.exists():
            print('Dataset exist!')
            return
    
        google_drive_link = 'https://drive.google.com/uc?id=1fY9IWK7yEfLOmS3wUgeXM3NIivhoGhsg'
        print(zip_file_path)
        gdown.download(google_drive_link, str(zip_file_path.absolute()), quiet=False)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)