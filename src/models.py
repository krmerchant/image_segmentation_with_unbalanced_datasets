import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu


class UNet(nn.Module):
    def __init__(self, number_classes):
        super().__init__()
        self.encoder1_1 = nn.Conv2d(3, 64, kernel_size=3)
        self.encoder1_2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2_1 = nn.Conv2d(64, 128, kernel_size=3)
        self.encoder2_2 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3_1 = nn.Conv2d(128, 256, kernel_size=3)
        self.encoder3_2 = nn.Conv2d(256, 256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4_1 = nn.Conv2d(256, 512, kernel_size=3)
        self.encoder4_2 = nn.Conv2d(512, 512, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5_1 = nn.Conv2d(512, 1024, kernel_size=3)
        self.encoder5_2 = nn.Conv2d(1024, 1024, kernel_size=3)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2)
        self.decoder1_1 = nn.Conv2d(1024, 512, kernel_size=3)
        self.decoder1_2 = nn.Conv2d(512, 512, kernel_size=3)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2)
        self.decoder2_1 = nn.Conv2d(512, 256, kernel_size=3)
        self.decoder2_2 = nn.Conv2d(256, 256, kernel_size=3)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2)
        self.decoder2_1 = nn.Conv2d(256, 128, kernel_size=3)
        self.decoder2_2 = nn.Conv2d(128, 128, kernel_size=3)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2)
        self.decoder4_1 = nn.Conv2d(128, 64, kernel_size=3)
        self.decoder4_2 = nn.Conv2d(64, 64, kernel_size=3)

        self.seg_out = nn.Conv2d(
            64, number_classes, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        """ forward pass of network"""
        
        #encoders
        x_e11 = nn.ReLU(self.encoder1_1(x))
        x_e12 = nn.ReLU(self.encoder1_2(x_e11))
        x_pool1 = self.pool1(x_e12)

        x_e21 = nn.ReLU(self.encoder2_1(x_pool1))
        x_e22 = nn.ReLU(self.encoder2_2(x_e21))
        x_pool2 = self.pool2(x_e22)

        x_e31 = nn.ReLU(self.encoder3_1(x_pool2))
        x_e32 = nn.ReLU(self.encoder3_2(x_e31))
        x_pool3 = self.pool3(x_e32)

        x_e41 = nn.ReLU(self.encoder4_1(x_pool3))
        x_e42 = nn.ReLU(self.encoder4_2(x_e41))
        x_pool4 = self.pool4(x_e42)

        x_e51 = nn.ReLU(self.encoder5_1(x_pool4))
        x_e52 = nn.ReLU(self.encoder5_2(x_e51))

        
        #decoder
        xup1 = self.upconv1(x_e52)
        xcc1 = torch.cat[xup1,x_e42]
        xd11 = nn.ReLU(self.decoder1_1(xcc1))
        xd12 = nn.ReLU(self.decoder1_2(xd11))

        xup2 = self.upconv2(xd12)
        xcc2 = torch.cat[xup2,x_e32]
        xd21 = nn.ReLU(self.decoder2_1(xcc2))
        xd22 = nn.ReLU(self.decoder2_2(xd21))
 
        xup3 = self.upconv3(xd22)
        xcc3 = torch.cat[xup3,x_e22]
        xd31 = nn.ReLU(self.decoder3_1(xcc3))
        xd32 = nn.ReLU(self.decoder3_2(xd31))

        xup4 = self.upconv4(xd32)
        xcc4 = torch.cat[xup4,x_e12]
        xd41 = nn.ReLU(self.decoder4_1(xcc4))
        xd42 = nn.ReLU(self.decoder4_2(xd41))

        output = self.seg_out(xd42)
        return output









