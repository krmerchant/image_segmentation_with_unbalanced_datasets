import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch.nn.functional as F 

class KittiDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.dataset_csv = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform


    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # pull index from csv file
        image_name = os.path.join(
            self.root_dir, "image_2/", self.dataset_csv.iloc[index, 0])
        seg_filename = os.path.join(
            self.root_dir, "semantic/", self.dataset_csv.iloc[index, 0])
        # normalize image from 0-1 off the bat
        return_image = torch.tensor(io.imread(image_name)/255, dtype=torch.float).permute(2, 0, 1)
        return_seg = torch.tensor(io.imread(seg_filename, as_gray=True), dtype=torch.int)
        return_seg = return_seg.long()
        return_seg = return_seg * (return_seg > 25) 

        # have to add make this 1xNxN to make torch.Resize() happy
        return_seg = return_seg.reshape(
            1, return_seg.shape[0], return_seg.shape[1])
        if (self.transform):
            return_image = self.transform(return_image)
            return_seg = self.transform(return_seg)
        one_hot_rep = F.one_hot(return_seg, 37)
        return return_image, one_hot_rep
