from models import UNet
import unittest
from datasets import KittiDataset
from torchvision import transforms, utils
import torch

class TestUNetNetwork(unittest.TestCase):

    def test_construction(self):
        unet = UNet(50)

  
    def test_pass_batch(self):
        """test to make sure we can pass in a batch and get an the expected value"""
        images = torch.ones(2,3,512,512)
        unet = UNet(37,debug=False)
        
        output = unet(images)
        b,c,w,h = output.shape
        self.assertEqual([2,37,512, 512], [b,c,w,h])



if __name__ == '__main__':
    unittest.main()
