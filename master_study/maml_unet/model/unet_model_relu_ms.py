# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
from torch.autograd import Variable

from .unet_parts import *

TEST = False

class UNet_Relu_MS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Relu_MS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up4 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up2 = up(256, 64)
        self.up1 = up(128, 64)
        self.outc = outconv(64, 1)
        
        # up{from which level}_top
        self.up5_top = topconv(512, 2**4) 
        self.up4_top = topconv(256, 2**3)
        self.up3_top = topconv(128, 2**2)
        self.up2_top = topconv(64, 2**1)
        
        self.ms_outc = outconv(5, out_channels)
        self.weight_layer = Variable(torch.rand(5), requires_grad=True)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_up4 = self.up4(x5, x4)
        x_up3 = self.up3(x_up4, x3)
        x_up2 = self.up2(x_up3, x2)
        x_up1 = self.up1(x_up2, x1)
        x_up0_1 = F.relu(self.outc(x_up1))
        
        # x_up{current level}_{from which level}
        x_up0_5 = self.up5_top(x5)
        x_up0_4 = self.up4_top(x_up4)
        x_up0_3 = self.up3_top(x_up3)
        x_up0_2 = self.up2_top(x_up2)
        
#         softmax_weight = F.softmax(self.weight_layer)
        
#         if TEST:
#             print(x5.shape)
#             print(x_up4.shape)
#             print(x_up3.shape)
#             print(x_up2.shape)
        
#             print(x_up0_1.shape)
#             print(x_up0_2.shape)
#             print(x_up0_3.shape)
#             print(x_up0_4.shape)
#             print(x_up0_5.shape)
            
#             print(softmax_weight)        
        
#         x_ms = F.relu(self.ms_outc(torch.cat([softmax_weight[0] * x_up0_1, 
#                                     softmax_weight[1] * x_up0_2,
#                                     softmax_weight[2] * x_up0_3, 
#                                     softmax_weight[3] * x_up0_4, 
#                                     softmax_weight[4] * x_up0_5], dim=1)))
        
        x_ms = F.relu(self.ms_outc( torch.cat([x_up0_1, 
                                   x_up0_2,
                                   x_up0_3, 
                                   x_up0_4, 
                                   x_up0_5], dim=1)))
        
#         if TEST:
#             print("finished")
#             exit()
        return x_ms