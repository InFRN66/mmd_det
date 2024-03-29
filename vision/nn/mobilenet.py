# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride): # depth wise
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



class MobileNetV1_vgg(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1_vgg, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride): # depth wise
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        # self.model = nn.Sequential(
        #     conv_bn(3, 32, 2),
        #     conv_dw(32, 64, 1),
        #     conv_dw(64, 128, 2),
        #     conv_dw(128, 128, 1),
        #     conv_dw(128, 256, 2),
        #     conv_dw(256, 256, 1),
        #     conv_dw(256, 512, 2),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 512, 1),
        #     conv_dw(512, 1024, 2),
        #     conv_dw(1024, 1024, 1),
        # )

        self.model = nn.Sequential(
            conv_bn(3, 64, 1),   #0
            conv_dw(64, 64, 1),  #1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_dw(64, 128, 1), #2
            conv_dw(128, 128, 1),#3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_dw(128, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            conv_dw(256, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1), 
            conv_dw(512, 512, 1), 
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),

            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )


        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x