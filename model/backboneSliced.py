from doctest import OutputChecker
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import os
import numpy as np

import sys 
sys.path.append("..")


class proposed_light_Lut(nn.Module):
    def __init__(self):
        super(proposed_light_Lut, self).__init__()
        
        self.w_3 = torch.tensor([
            [2**8, 2**7, 2**6],
            [2**5, 2**4, 2**3],
            [2**2, 2**1, 2**0]
        ]).unsqueeze(0).unsqueeze(0).to(torch.float32)

        self.w_5 = torch.tensor([
            [2**24, 2**23, 2**22, 2**21, 2**20],
            [2**19, 2**18, 2**17, 2**16, 2**15],
            [2**14, 2**13, 2**12, 2**11, 2**10],
            [2**9,  2**8,  2**7,  2**6,  2**5],
            [2**4,  2**3,  2**2,  2**1,  2**0]
        ]).unsqueeze(0).unsqueeze(0).to(torch.float64)

        # lut backbone
        self.conv_2_5_0 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16), nn.LeakyReLU(negative_slope=0.01),
                torch.nn.Conv2d(16, 32, kernel_size=1, stride=1),
                nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.01),
                torch.nn.Conv2d(32, 64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01)
                )
        
    
        self.conv_down_to_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01))
        
        self.conv_down_to_8 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01))
        
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01))
        
        self.conv_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01))
        
        self.conv_8 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, dilation=2, padding=4),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.01))

        self.leakRelu = nn.LeakyReLU()
        
    def forward(self, x, lut_5=None, w5=None):   
        if lut_5 is None:
            # Network
            x = F.pad(x, (2, 2, 2, 2), value=0).to(torch.float32)
            x_2_5_0 = self.conv_2_5_0(x)
            x_2_5 = x_2_5_0[:,:,1:353,1:641]
        else:
            # BE-LUT
            x = F.pad(x, (2, 2, 2, 2), mode="constant", value=0)
            x_5_idx = F.conv2d(x.to(torch.float64), w5.to(torch.float64), stride=2).to(torch.int64).squeeze(0).squeeze(0)
            x_2_5 = lut_5[:, :, x_5_idx]

        
        # SCE
        x_4_down = self.conv_down_to_4(x_2_5)
        x_8_down = self.conv_down_to_8(x_4_down)
        
        x_8_down_conv = self.conv_8(x_8_down)
        x_4 = F.interpolate(x_8_down_conv, scale_factor=2, mode='nearest')
        
        x_4_down_conv = self.conv_4(self.leakRelu(x_4 + x_4_down))
        x_2 = F.interpolate(x_4_down_conv, scale_factor=2, mode='nearest')

        x_2_add = torch.add(x_2, x_2_5)
        x_2_down_conv = self.conv_2(self.leakRelu(x_2_add))
        output = self.leakRelu(torch.add(x_2_down_conv, x_2_add))
        
        return output
