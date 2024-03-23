import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class KittiDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.dataset_csv = pd.read_csv(os.path.join(self.root_dir,csv_file))    
        self.tranform = transform

    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # pull index from csv file
        image_name = os.path.join(self.root_dir, "image_2/", self.dataset_csv.iloc[index,0]);
        seg_filename = os.path.join(self.root_dir, "semantic/", self.dataset_csv.iloc[index,0]);
        seg_rgb_filename = os.path.join(self.root_dir, "semantic_rgb/", self.dataset_csv.iloc[index,0]);
        image = io.imread(image_name)
        seg_image = io.imread(seg_filename)
        seg_rgb_image = io.imread(seg_rgb_filename)
        sample = {'image': image, 'seg_mask':seg_image, 'seg_rgb_mask':seg_rgb_image}

        if(self.tranform):
            sample = self.tranform(sample)
        
        return sample
    


