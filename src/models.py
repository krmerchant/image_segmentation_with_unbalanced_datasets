import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu


def center_crop(x, N):
    """Extract Center"""
    c, w, h = x.shape
    return x[:, w//2-(N//2):w//2+(N//2), h//2-(N//2):h//2+(N//2)]


class UNet(nn.Module):
    def __init__(self, number_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.encoder1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) #3x572x572
        self.encoder1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) #64x572x572 (this concats accross)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #64x286x286

        self.encoder2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #128x256x256
        self.encoder2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) #128x256x256
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #128x128x128

        self.encoder3_1 = nn.Conv2d(128, 256, kernel_size=3,padding=1) #256X128x128
        self.encoder3_2 = nn.Conv2d(256, 256, kernel_size=3,padding=1) #256X128x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #256x64x64

        self.encoder4_1 = nn.Conv2d(256, 512, kernel_size=3,padding=1) #512x64x64
        self.encoder4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) #512x64x64
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #512x32x32

        self.encoder5_1 = nn.Conv2d(512, 1024, kernel_size=3,padding=1) #1024x32x32
        self.encoder5_2 = nn.Conv2d(1024, 1024, kernel_size=3,padding=1) #1024x64x64

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1_1 = nn.Conv2d(1024, 512, kernel_size=3,padding=1)
        self.decoder1_2 = nn.Conv2d(512, 512, kernel_size=3,padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2_1 = nn.Conv2d(512, 256, kernel_size=3,padding=1)
        self.decoder2_2 = nn.Conv2d(256, 256, kernel_size=3,padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3_1 = nn.Conv2d(256, 128, kernel_size=3,padding=1)
        self.decoder3_2 = nn.Conv2d(128, 128, kernel_size=3,padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4_1 = nn.Conv2d(128, 64, kernel_size=3,padding=1)
        self.decoder4_2 = nn.Conv2d(64, 64, kernel_size=3,padding=1)

        self.seg_out = nn.Conv2d(
            64, number_classes, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        """ forward pass of network"""

        # encoders
        x_e11 = relu(self.encoder1_1(x))
        if self.debug: print(f"{x_e11.shape=}")
        x_e12 = relu(self.encoder1_2(x_e11))
        if self.debug: print(f"{x_e12.shape=}")
        x_pool1 = self.pool1(x_e12)
        if self.debug: print(f"{x_pool1.shape=}")
        x_e21 = relu(self.encoder2_1(x_pool1))
        if self.debug: print(f"{x_e21.shape=}")
        x_e22 = relu(self.encoder2_2(x_e21))
        if self.debug: print(f"{x_e22.shape=}")
        x_pool2 = self.pool2(x_e22)
        if self.debug: print(f"{x_pool2.shape=}")
        
        x_e31 = relu(self.encoder3_1(x_pool2))
        if self.debug: print(f"{x_e31.shape=}")
        x_e32 = relu(self.encoder3_2(x_e31))
        if self.debug: print(f"{x_e32.shape=}")
        x_pool3 = self.pool3(x_e32)
        if self.debug: print(f"{x_pool3.shape=}")
        
        x_e41 = relu(self.encoder4_1(x_pool3))
        x_e42 = relu(self.encoder4_2(x_e41))
        x_pool4 = self.pool4(x_e42)

        x_e51 = relu(self.encoder5_1(x_pool4))
        x_e52 = relu(self.encoder5_2(x_e51))

        # decoder

        if self.debug: print(f"{x_e52.shape=}")
        xup1 = self.upconv1(x_e52)
        if self.debug: print(f"{xup1.shape=}")
        xcc1 = torch.cat([xup1, x_e42], dim=0)
        if self.debug: print(f"{xcc1.shape=}")
        xd11 = relu(self.decoder1_1(xcc1))
        xd12 = relu(self.decoder1_2(xd11))

        xup2 = self.upconv2(xd12)
        xcc2 = torch.cat([xup2, x_e32], dim=0)
        xd21 = relu(self.decoder2_1(xcc2))
        xd22 = relu(self.decoder2_2(xd21))

        xup3 = self.upconv3(xd22)
        xcc3 = torch.cat([xup3, x_e22], dim=0)
        xd31 = relu(self.decoder3_1(xcc3))
        xd32 = relu(self.decoder3_2(xd31))

        xup4 = self.upconv4(xd32)
        xcc4 = torch.cat([xup4, x_e12], dim=0)
        xd41 = relu(self.decoder4_1(xcc4))
        xd42 = relu(self.decoder4_2(xd41))

        output = self.seg_out(xd42)
        return output
