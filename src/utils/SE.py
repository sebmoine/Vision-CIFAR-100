import torch
import torch.nn as nn

# Squeeze-and-Excitation is a building block for CNNs that improves channel interdependencies at almost no computational cost.
# When first used on ImageNet competition, it improves the result from the previous competition by 25%.
# The purpose is to add parameters to each channel of a convolutional block so that the network can adaptively adjust the weighting of each feature map

# From "the network weights each of its channels equally when creating the output feature maps" To "content aware mechanism to weight each channel adaptively"


# source : Medium - Squeeze-and-Excitation Networks (https://medium.com/data-science/squeeze-and-excitation-networks-9ef5e71eacd7)

class SE_Block(nn.Module):
    def __init__(self, out_channels, ratio=16): # an input convolutional block and the current number of channels it has
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)                      # B x C x 1 x 1
        self.flatten = nn.Flatten(1,-1)                             # B x C
        self.dense1 = nn.Linear(out_channels, out_channels//ratio)  # B x C/r
        self.relu    = nn.ReLU(True)                                # B x C/r
        self.dense2 = nn.Linear(out_channels//ratio, out_channels)  # B x C
        self.sigmoid = nn.Sigmoid()                                 # B x C

    def forward(self, in_block):
        x = self.avgpool(in_block)      # squeeze each channel to a single numeric value
        x = self.flatten(x)
        x = self.dense1(x)              # reduce output channel complexity divided by a ratio
        x = self.relu(x)                # nonlinearity
        x = self.dense2(x)
        x = self.sigmoid(x)             # gives each channel a smooth gating function
        x = x.view(x.size(0), x.size(1), 1, 1)
        return torch.mul(x, in_block)   # weights each feature map of the convolutional block (in_block) based on the result of our x, a vector of size n=nb of filters in the conv. block.
        # W x H x C