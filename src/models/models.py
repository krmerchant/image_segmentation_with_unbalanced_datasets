import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu


class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionBlock,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class DoubleConv(nn.Module):
    def __init__(self, input_channels, out_channels, debug=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels), 
            nn.ReLU(), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels), 
            nn.ReLU() ,
        )
    def forward(self,x):
        """forwared"""
        output = self.double_conv(x)
        return output

class AttentionUNet(nn.Module):
    def __init__(self, number_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.number_classes = number_classes
        self.encoder1 = DoubleConv(3,64) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #64x286x286


        self.encoder2 = DoubleConv(64,128) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #128x128x128

        self.encoder3 = DoubleConv(128, 256) #256X128x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #256x64x64

        self.encoder4 = DoubleConv(256, 512) #512x64x64
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #512x32x32

        self.encoder5 = DoubleConv(512, 1024) #512x64x64

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=512, F_l=512, F_int=256)  
        self.decoder1 = DoubleConv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=256, F_l=256, F_int=128)  
        self.decoder2 =  DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder3 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention4  = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder4 = DoubleConv(128, 64)

        self.seg_out = nn.Conv2d(
            64, number_classes, kernel_size=1)  # 1x1 convolution
        if(number_classes == 1):
            self.sig_out = nn.Sigmoid()
        else:
            self.soft_max_out = nn.Softmax2d()
    def forward(self, x):
        """ forward pass of network"""

        # encoders
        x_e1 = self.encoder1(x) 
        x_pool1 = self.pool1(x_e1)
        if self.debug: print(f"{x_pool1.shape=}")
        
        x_e2 = self.encoder2(x_pool1)
        if self.debug: print(f"{x_e2.shape=}")
        x_pool2 = self.pool2(x_e2)
        if self.debug: print(f"{x_pool2.shape=}")
        
        x_e3 = self.encoder3(x_pool2)
        if self.debug: print(f"{x_e3.shape=}")
        x_pool3 = self.pool3(x_e3)
        if self.debug: print(f"{x_pool3.shape=}")
        
        x_e4 = self.encoder4(x_pool3) 
        x_pool4 = self.pool4(x_e4)

        x_e5 = self.encoder5(x_pool4) 

        xup1 = self.upconv1(x_e5)
        xatt1 = self.attention1(g=xup1, x=x_e4) #output of attention block  
        xcc1 = torch.cat([xup1,xatt1], dim=1)
        xd1 = self.decoder1(xcc1); 

        xup2 = self.upconv2(xd1)
        xatt2 = self.attention2(g=xup2, x=x_e3) #output of attention block  
        xcc2 = torch.cat([xup2, xatt2], dim=1)
        xd2 = self.decoder2(xcc2)

        xup3 = self.upconv3(xd2)
        xatt3 = self.attention3(g=xup3, x=x_e2) #output of attention block  
        xcc3 = torch.cat([xup3, xatt3], dim=1)
        xd3 =  self.decoder3(xcc3) 

        xup4 = self.upconv4(xd3)
        xatt4 = self.attention4(g=xup4, x=x_e1) #output of attention block  
        xcc4 = torch.cat([xup4, xatt4], dim=1)
        xd4 = self.decoder4(xcc4)
        output = self.seg_out(xd4)
        #clamp to prob 
        if(self.number_classes == 1):
            if self.debug: print("output")
            output = self.sig_out(output)
        else:
            output = self.soft_max_out(output)

        return output



class UNet(nn.Module):
    def __init__(self, number_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.number_classes = number_classes
        self.encoder1 = DoubleConv(3,64) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #64x286x286


        self.encoder2 = DoubleConv(64,128) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #128x128x128

        self.encoder3 = DoubleConv(128, 256) #256X128x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #256x64x64

        self.encoder4 = DoubleConv(256, 512) #512x64x64
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #512x32x32

        self.encoder5 = DoubleConv(512, 1024) #512x64x64

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 =  DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(128, 64)

        self.seg_out = nn.Conv2d(
            64, number_classes, kernel_size=1)  # 1x1 convolution
        if(number_classes == 1):
            self.sig_out = nn.Sigmoid()
        else:
            self.soft_max_out = nn.Softmax2d()
    def forward(self, x):
        """ forward pass of network"""

        # encoders
        x_e1 = self.encoder1(x) 
        x_pool1 = self.pool1(x_e1)
        if self.debug: print(f"{x_pool1.shape=}")
        
        x_e2 = self.encoder2(x_pool1)
        if self.debug: print(f"{x_e2.shape=}")
        x_pool2 = self.pool2(x_e2)
        if self.debug: print(f"{x_pool2.shape=}")
        
        x_e3 = self.encoder3(x_pool2)
        if self.debug: print(f"{x_e3.shape=}")
        x_pool3 = self.pool3(x_e3)
        if self.debug: print(f"{x_pool3.shape=}")
        
        x_e4 = self.encoder4(x_pool3) 
        x_pool4 = self.pool4(x_e4)

        x_e5 = self.encoder5(x_pool4) 
        # decoderrelu(self.encoder5_1(x_pool4))


        if self.debug: print(f"{x_e5.shape=}")
        xup1 = self.upconv1(x_e5)
        if self.debug: print(f"{xup1.shape=}")
        xcc1 = torch.cat([xup1, x_e4], dim=1)
        if self.debug: print(f"{xcc1.shape=}")
        xd1 = self.decoder1(xcc1); 
        if self.debug: print(f"{xd1.shape=}")

        xup2 = self.upconv2(xd1)
        xcc2 = torch.cat([xup2, x_e3], dim=1)
        xd2 = self.decoder2(xcc2)

        xup3 = self.upconv3(xd2)
        xcc3 = torch.cat([xup3, x_e2], dim=1)
        xd3 =  self.decoder3(xcc3) 

        xup4 = self.upconv4(xd3)
        xcc4 = torch.cat([xup4, x_e1], dim=1)
        xd4 = self.decoder4(xcc4)
        output = self.seg_out(xd4)
        #clamp to prob 
        if(self.number_classes == 1):
            if self.debug: print("output")
            output = self.sig_out(output)
        else:
            output = self.soft_max_out(output)

        return output
