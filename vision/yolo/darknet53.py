import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Bn_Leaky(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        super(Conv_Bn_Leaky, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.leaky_relu(out, negative_slope=0.1, inplace=True)


class DarkNetBlock(nn.Module):
    def __init__(self, ch1, ch2, skip=True):
        super(DarkNetBlock, self).__init__()
        self.conv1 = Conv_Bn_Leaky(ch1, ch2, ksize=1, stride=1, pad=0)
        self.conv2 = Conv_Bn_Leaky(ch2, ch1, ksize=3, stride=1, pad=1)
        self.skip = skip

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.skip:
            return out + residual
        else:
            return out

class DarkNet53(nn.Module):
    def __init__(self, blocks): # blocks = [1, 2, 8, 8, 4]
        super(DarkNet53, self).__init__()
        self.blocks = blocks
        self.conv = Conv_Bn_Leaky(3, 32, 3, 1, 1)
        self.block1 = self.create_block(blocks[0], 32, 64)
        self.block2 = self.create_block(blocks[1], 64, 128)
        self.block3 = self.create_block(blocks[2], 128, 256)
        self.block4 = self.create_block(blocks[3], 256, 512)
        self.block5 = self.create_block(blocks[4], 512, 1024)

    def create_block(self, num, in_ch, out_ch):
        layers = []
        # downsample
        layers.append(Conv_Bn_Leaky(in_ch, out_ch, 3, 2, 1)) # 32 - 64
        for _ in range(num):
            layers.append(DarkNetBlock(out_ch, in_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        extract = list()
        out = self.conv(x)  # (32, 256)
        out = self.block1(out) # (64, 128)
        out = self.block2(out)
        out = self.block3(out)
        extract.append(out)
        out = self.block4(out)
        extract.append(out)
        out = self.block5(out)
        extract.append(out)
        return extract
        
