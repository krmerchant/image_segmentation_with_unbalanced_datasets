from models import UNet
import unittest
from datasets import KittiDataset
from torchvision import transforms, utils
import torch
import torch.nn as nn

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

    def test_loss_computation(self):
        '''Just test that we can pass in a batch-like torch array'''
        unet = UNet(37,debug=False)
        criterion =  nn.CrossEntropyLoss()
        images = torch.ones(2,3,512,512)
        labels = torch.ones(2,512,512).long()

        output = unet(images)
        loss = criterion(output,labels) 
        print(loss.item())
        self.assertIsNotNone(loss.item()) ##is the loss of the right dims


if __name__ == '__main__':
    unittest.main()
