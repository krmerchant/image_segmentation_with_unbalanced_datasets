import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.util import crop
class KittiDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.dataset_csv = pd.read_csv(os.path.join(self.root_dir,csv_file))    
        self.tranform = transform
        self.tranform = transform

    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # pull index from csv file
        image_name = os.path.join(self.root_dir, "image_2/", self.dataset_csv.iloc[index,0]);
        seg_filename = os.path.join(self.root_dir, "semantic/", self.dataset_csv.iloc[index,0]);
        return_image = torch.Tensor(io.imread(image_name)/255).permute(2,0,1) #normalize image from 0-1 off the bat 
        return_seg = torch.Tensor(io.imread(seg_filename))
        if(self.tranform):
            return_image = self.tranform(return_image)
            return_seg = self.tranform(return_seg)
        
        return return_image,return_seg
    

