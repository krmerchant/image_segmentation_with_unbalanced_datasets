from models import UNet
from models import AttentionUNet
import unittest
from datasets import KittiDataset
from torchvision import transforms, utils
import torch
import torch.nn as nn

class TestUNetNetwork(unittest.TestCase):

    def test_construction(self):
        unet = UNet(50)






 
    def test_pass_batch_multiclass_attention_unet(self):
        """test to make sure we can pass in a batch and get an the expected value"""
        images = torch.ones(2,3,512,512)
        unet = AttentionUNet(34,debug=False)
        output = unet(images)
        b,c,w,h = output.shape
        self.assertEqual([2,34,512, 512], [b,c,w,h])


    def test_pass_batch_multiclass(self):
        """test to make sure we can pass in a batch and get an the expected value"""
        images = torch.ones(2,3,512,512)
        unet = UNet(34,debug=False)
        
        output = unet(images)
        b,c,w,h = output.shape
        self.assertEqual([2,34,512, 512], [b,c,w,h])


    def test_pass_batch(self):
        """test to make sure we can pass in a batch and get an the expected value"""
        images = torch.ones(2,3,512,512)
        unet = UNet(37,debug=False)
        
        output = unet(images)
        b,c,w,h = output.shape
        self.assertEqual([2,37,512, 512], [b,c,w,h])

    def test_loss_computation(self):
        '''Just test that we can pass in a batch-like torch array'''
        unet = UNet(2,debug=False)
        criterion =  nn.CrossEntropyLoss()
        images = torch.ones(2,3,512,512)
        labels = torch.ones(2,512,512).long()

        output = unet(images)
        print(output)
        loss = criterion(output,labels) 
        print(loss.item())
        self.assertIsNotNone(loss.item()) ##is the loss of the right dims
    def test_loss_computation_binary_model(self):
        '''Just test that we can pass in a batch-like torch array'''
        unet = UNet(1,debug=False)
        criterion =  nn.CrossEntropyLoss()
        images = torch.zeros(2,3,512,512)
        labels = torch.ones(2,1,512,512).float()
        
        output = unet(images)
        print(output)
        loss = criterion(output,labels) 
        print(loss.item())
        self.assertIsNotNone(loss.item()) ##is the loss of the right dims



if __name__ == '__main__':
    unittest.main()
