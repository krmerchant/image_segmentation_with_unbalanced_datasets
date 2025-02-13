from datasets import KittiDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import torch
import unittest
import pytest
import torch.nn.functional as F

class TestKittiDatasetLoader(unittest.TestCase):

    def test_basic_dataset(self):
        kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training',car_only=False, transform=transforms.Compose([transforms.Resize((512, 512))])) 
        for (i, (image, seg)) in enumerate(kitti):
            #if i == 100:
            #    print(image)
            #    numpy_image = image.permute(1, 2, 0).numpy()
            #    numpy_seg = seg.reshape(seg.shape[1], seg.shape[2]).numpy()
            #    print(numpy_seg.shape)
            #    fig, ax = plt.subplots(1, 2)
            #    ax[0].imshow((numpy_image))
            #    seg_map = ax[1].imshow(numpy_seg)
            #    fig.colorbar(seg_map)
            #    print(numpy_seg)
            c, h, w = image.shape
            c_seg, h_seg, w_seg = seg.shape
            self.assertEqual([3,512,512], [c, h, w])
            self.assertEqual([1,512,512], [c_seg, h_seg, w_seg])

    def test_data_loader_multi_lable(self):        
        kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training',car_only=False, transform=transforms.Compose([transforms.Resize((512, 512))])) 
        unique_labels = torch.tensor([])
        data_loader = DataLoader(kitti, batch_size=2, shuffle=True)
        for batch in data_loader:
            inputs, labels = batch  # Assuming each batch returns inputs and labels
            unique_labels = torch.cat([torch.unique(labels), unique_labels],0);
            unique_labels = torch.unique(unique_labels)

        self.assertEqual(unique_labels.shape[0], 34)
           # Your training/evaluation loop
        print("Sucessfully interated through batches!")


    def test_one_hot_batch(self):
        kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training',car_only=False, transform=transforms.Compose([transforms.Resize((512, 512))])) 

        data_loader = DataLoader(kitti, batch_size=2, shuffle=True)
        for batch in data_loader:
            _, labels = batch
            labels = F.one_hot(labels, num_classes=34)
            labels = torch.squeeze(labels, dim=1)
            [b,w,h,c]  = labels.shape 
            self.assertEqual([512,512,34], [w,h,c])
if __name__ == "__main__":
    unittest.main()
