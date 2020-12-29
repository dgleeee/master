
import torch.nn as nn
import numpy as np
""" Optional conv block """
def conv_block1(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def conv_block2(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def conv_block3(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.AvgPool2d(2),
    )

""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel, self).__init__()       
        
        self.conv1 = conv_block1(x_dim, hid_dim)                     #conv_block(in_channels, out_channels) : [conv layer, batch norm, Relu 활성화 함수] 3개 반복, 입력은 input과 output의 channel 개수
        self.conv2 = conv_block1(hid_dim, hid_dim)
        self.conv4 = conv_block1(hid_dim, z_dim)
        self.conv5 = conv_block1(z_dim, z_dim)
        
        self.conv6 =  conv_block2(z_dim, z_dim)
        self.conv7 =  conv_block3(z_dim, z_dim)
        self.pool = nn.MaxPool2d(2)
        self.skip = nn.Identity()
        
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(z_dim, z_dim, 3, padding=1),
        #     nn.BatchNorm2d(z_dim),
        #     nn.ReLU(),
        #     )

        # self.dim_equalizer1 = nn.Conv2d(x_dim, hid_dim, 3, padding=1)
        # self.dim_equalizer2 = nn.Conv2d(hid_dim,z_dim,kernel_size=1)
        # self.avgpool = nn.AvgPool2d(2)
##==========================================================================


    def forward(self, x):
        out1 = self.conv1(x)
        # out1 = out1 + self.skip(self.pool(self.dim_equalizer1(x)))
        
        out2 = self.conv2(out1) + out1
#        out2 = out2 + self.skip(self.pool(out1))
        
        out3 = self.conv2(out2) + out2
#        out3 = out3 + self.skip(self.pool(out2))
        
        out4 = self.conv4(out3) + out3
#        out4 = out4 + self.skip(self.pool(out3))
        # out4 = out4 + self.skip(self.pool(self.dim_equalizer2(out3)))

        out5 = self.conv5(out4) + out4
#        out5 = out5 + self.skip(self.pool(out4))
        
        out6 = self.conv5(out5) + out5
#        out6 = out6 + self.skip(self.pool(out5))
        
#        out7 = self.conv5(out6)
#        out7 = out7 + self.skip(self.pool(out6))

        out = self.conv7(out6) + out6
        # out = out + self.skip(self.pool(out7))
        
        embedding_vector = out.view(out.size(0), -1)
        return embedding_vector
