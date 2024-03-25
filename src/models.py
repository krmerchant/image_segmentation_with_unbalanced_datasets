import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu



class UNet(nn.Module):
    def __init__(self, number_classes):
        super().__init__()
        self.encoder1_1 = nn.Conv2d(3,64,kernel_size=3)
        self.encoder1_2 = nn.Conv2d(64,64,kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
    
        self.encoder2_1 = nn.Conv2d(64,128,kernel_size=3)
        self.encoder2_2 = nn.Conv2d(128,128,kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
    
        self.encoder3_1 = nn.Conv2d(128,256,kernel_size=3)
        self.encoder3_2 = nn.Conv2d(256,256,kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
    
        self.encoder4_1 = nn.Conv2d(256,512,kernel_size=3)
        self.encoder4_2 = nn.Conv2d(512,512,kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
    
        self.encoder5_1 = nn.Conv2d(512,1024,kernel_size=3)
        self.encoder5_2 = nn.Conv2d(1024,1024,kernel_size=3)
    
        self.upconv1  = nn.ConvTranspose2d(1024,512,kernel_size=2)
        self.decoder1_1 = nn.Conv2d(1024,512,kernel_size=3)
        self.decoder1_2 = nn.Conv2d(512,512,kernel_size=3)
    
        self.upconv2 = nn.ConvTranspose2d(512,256,kernel_size=2)
        self.decoder2_1 = nn.Conv2d(512,256,kernel_size=3)
        self.decoder2_2 = nn.Conv2d(256,256,kernel_size=3)
    
        self.upconv3 = nn.ConvTranspose2d(256,128,kernel_size=2)
        self.decoder2_1 = nn.Conv2d(256,128,kernel_size=3)
        self.decoder2_2 = nn.Conv2d(128,128,kernel_size=3)
    
        self.upconv4 = nn.ConvTranspose2d(128,64,kernel_size=2)
        self.decoder4_1 = nn.Conv2d(128,64,kernel_size=3)
        self.decoder4_2 = nn.Conv2d(64,64,kernel_size=3)
    
        self.decoder4_2 = nn.Conv2d(64,number_classes,kernel_size=1) #1x1 convolution
    
    
    
    
    