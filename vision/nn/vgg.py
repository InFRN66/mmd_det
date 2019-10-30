import numpy as np
import torch
import torch.nn as nn


# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
def vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else: # number
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# --- custom function
def vgg_fractionalPooling(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M': # ordinary max pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "F": # fractional max pooling
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_ratio=1/(np.sqrt(2)-0.05))] # ratio
        elif v == "F11": # fractional max pooling
            size = int(np.ceil(300*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-1
        elif v == "F12": # fractional max pooling
            size = int(np.ceil(300*(1/np.sqrt(2))*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-2
        elif v == "F21": # fractional max pooling
            size = int(np.ceil(150*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-1
        elif v == "F22": # fractional max pooling
            size = int(np.ceil(150*(1/np.sqrt(2))*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-2
        elif v == "F31": # fractional max pooling
            size = int(np.ceil(75*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-1
        elif v == "F32": # fractional max pooling
            size = int(np.ceil(75*(1/np.sqrt(2))*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-2
        elif v == "F41": # fractional max pooling
            size = int(np.ceil(38*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-1
        elif v == "F42": # fractional max pooling
            size = int(np.ceil(38*(1/np.sqrt(2))*(1/np.sqrt(2))))
            layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 38-2
        elif v == 'C': # ceil mode max pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else: # convolution + bn + RELU
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # can't convert to fraction lapooling, because not subsampling 
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    # # --- fractionalP for 19
    # size = int(np.ceil(19*(1/np.sqrt(2))))
    # layers += [nn.FractionalMaxPool2d(kernel_size=2, output_size=(size, size))] # 19-1]
    return layers