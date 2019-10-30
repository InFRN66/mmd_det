import torch
import torch.nn as nn
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.vgg import vgg, vgg_fractionalPooling

from .ssd import SSD, SSD_complement_with_fractional
from .predictor import Predictor
# from .config.vgg_ssd_config import Config

# custom func
def create_vgg_ssd(num_classes, config, is_test=False):
    num_anchors = config.num_anchors

    vgg_config = config.vgg_config
    
    # base_net = ModuleList(vgg(vgg_config)) # == ModuleList([Conv, Conv, Conv, ReLU, ,....]) / conv1 ~ conv5 + conv6, conv7
    base_net = ModuleList(vgg_fractionalPooling(vgg_config)) # == ModuleList([Conv, Conv, Conv, ReLU, ,....]) / conv1 ~ conv5 + conv6, conv7
    
    if config.pretrained:
        base_net = load_pre_weights(base_net, config)

    if config.vgg_config == [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]:
        print("vgg_config: basic.")
        source_layer_indexes = [
            (23, BatchNorm2d(512)),
             len(base_net),
        ]
    elif config.vgg_config == [64, 64, "F11", 64, "F12", 128, 128, "F21", 128, "F22", 256, 256, 256, "F31", 256, "F32",
                               512, 512, 512, "F41", 512, "F42", 512, 512, 512]:
        print("vgg_config: fracPool w/ conv+ReLU.")
        source_layer_indexes = [
            (32, BatchNorm2d(512)),
             len(base_net),
        ]
    elif config.vgg_config == [64, 64, "F11", "F12", 128, 128, "F21", "F22", 256, 256, 256, "F31", "F32",
                               512, 512, 512, "F41", "F42", 512, 512, 512]:
        print("vgg_config: fracPool w/o conv+ReLU.")
        source_layer_indexes = [
            (26, BatchNorm2d(512)),
             len(base_net),
        ]

    # for fractionalP
    # source_layer_indexes = [
    #     (23, BatchNorm2d(512)),
    #     24,
    #     len(base_net)-1,
    #     len(base_net)
    # ]

    extras = ModuleList([
        # conv8
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv9
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv10
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        # conv11
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * 4, kernel_size=3, padding=1), 
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * 4, kernel_size=3, padding=1), # 14
        Conv2d(in_channels=512, out_channels=num_anchors[2] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)



def create_vgg_ssd_complement_with_fractional(num_classes, config, is_test=False):
    num_anchors = config.num_anchors
    vgg_config = config.vgg_config
    # base_net = ModuleList(vgg(vgg_config)) # == ModuleList([Conv, Conv, Conv, ReLU, ,....]) / conv1 ~ conv5 + conv6, conv7
    base_net = ModuleList(vgg_fractionalPooling(vgg_config)) # == ModuleList([Conv, Conv, Conv, ReLU, ,....]) / conv1 ~ conv5 + conv6, conv7
    
    if config.pretrained:
        base_net = load_pre_weights(base_net, config)

    if config.vgg_config == [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]:
        print("vgg_config: basic.")
        source_layer_indexes = [
            (23, BatchNorm2d(512)),
             len(base_net),
        ]
    elif config.vgg_config == [64, 64, "F11", 64, "F12", 128, 128, "F21", 128, "F22", 256, 256, 256, "F31", 256, "F32",
                               512, 512, 512, "F41", 512, "F42", 512, 512, 512]:
        print("vgg_config: fracPool w/ conv+ReLU.")
        source_layer_indexes = [
            (32, BatchNorm2d(512)),
             len(base_net),
        ]
    elif config.vgg_config == [64, 64, "F11", "F12", 128, 128, "F21", "F22", 256, 256, 256, "F31", "F32",
                               512, 512, 512, "F41", "F42", 512, 512, 512]:
        print("vgg_config: fracPool w/o conv+ReLU.")
        source_layer_indexes = [
            (26, BatchNorm2d(512)),
             len(base_net),
        ]

    # for fractionalP
    # source_layer_indexes = [
    #     (23, BatchNorm2d(512)),
    #     24,
    #     len(base_net)-1,
    #     len(base_net)
    # ]

    extras = ModuleList([
        # conv8
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv9
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv10
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        # conv11
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    complement_fractional = nn.ModuleList([
        # 38 -> 34
        Sequential(
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.FractionalMaxPool2d(kernel_size=2, output_size=28)
        ),
        # 19 -> 14
        Sequential(
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.FractionalMaxPool2d(kernel_size=2, output_size=15)
        ),
        # 10 -> 7
        Sequential(
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.FractionalMaxPool2d(kernel_size=2, output_size=7)
        ),
        # 5 -> 4
        Sequential(
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.FractionalMaxPool2d(kernel_size=2, output_size=4)
        ),
        # 3 -> 2
        Sequential(
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.FractionalMaxPool2d(kernel_size=2, output_size=2)
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * 4, kernel_size=3, padding=1),  # 38 (out_size)
        Conv2d(in_channels=512, out_channels=num_anchors[0] * 4, kernel_size=3, padding=1),  # 28
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * 4, kernel_size=3, padding=1), # 19
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * 4, kernel_size=3, padding=1), # 15
        Conv2d(in_channels=512, out_channels=num_anchors[2] * 4, kernel_size=3, padding=1),  # 10
        Conv2d(in_channels=512, out_channels=num_anchors[2] * 4, kernel_size=3, padding=1),  # 7
        Conv2d(in_channels=256, out_channels=num_anchors[3] * 4, kernel_size=3, padding=1),  # 5
        Conv2d(in_channels=256, out_channels=num_anchors[3] * 4, kernel_size=3, padding=1),  # 4
        Conv2d(in_channels=256, out_channels=num_anchors[4] * 4, kernel_size=3, padding=1),  # 3
        Conv2d(in_channels=256, out_channels=num_anchors[4] * 4, kernel_size=3, padding=1),  # 2
        Conv2d(in_channels=256, out_channels=num_anchors[5] * 4, kernel_size=3, padding=1),  # 1
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[0] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * num_classes, kernel_size=3, padding=1),
    ])

    return SSD_complement_with_fractional(num_classes, base_net, source_layer_indexes, 
                                          extras, complement_fractional, classification_headers, regression_headers,
                                          is_test=is_test, config=config)


# ordinary
def create_vgg_ssd_old(num_classes, anchor_status, is_test=False):
    '''
    anchor status: list of num anchors in each layers
                    default [4,6,6,6,4,4] 
    '''
    vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                  512, 512, 512]
    base_net = ModuleList(vgg(vgg_config)) # == ModuleList([Conv, Conv, Conv, ReLU, ,....]) / conv1 ~ conv5 + conv6, conv7

    source_layer_indexes = [
        (23, BatchNorm2d(512)),
        len(base_net),
    ]
    extras = ModuleList([
        # conv8
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv9
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv10
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        # conv11
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)



def create_vgg_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None, config=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold, # NMS用
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor


# --- load part of pretrained weight
def load_pre_weights(base_net, config):
    weight = torch.load(config.pretrained)
    weight_dict = get_weight_match(weight, config)
    for i, layer in enumerate(base_net):
        if i in list(weight_dict["weight"].keys()):    
            layer.weight = nn.Parameter(weight_dict["weight"][i])
            layer.bias   = nn.Parameter(weight_dict["bias"][i])
        else:
            continue
    return base_net

# --- 
def get_weight_match(weight, config):
    if config.vgg_config == [64, 64, "F11", 64, "F12", 128, 128, "F21", 128, "F22", 256, 256, 256, "F31", 256, "F32",
                             512, 512, 512, "F41", 512, "F42", 512, 512, 512]:
        print("load weight for fracPool w/ conv+ReLU.")
        weight_dict = \
                {"weight":
                        {
                            0: weight["0.weight"],
                            2: weight["2.weight"], # conv1
                            8: weight["5.weight"],
                            10: weight["7.weight"], # conv2
                            16: weight["10.weight"],
                            18: weight["12.weight"],
                            20: weight["14.weight"], # conv3
                            26: weight["17.weight"],
                            28: weight["19.weight"],
                            30: weight["21.weight"], # conv4
                            36: weight["24.weight"],
                            38: weight["26.weight"],
                            40: weight["28.weight"], # conv5
                            43: weight["31.weight"], # conv6
                            45: weight["33.weight"], # conv7
                        },
                "bias":
                        {
                            0: weight["0.bias"],
                            2: weight["2.bias"], # conv1
                            8: weight["5.bias"],
                            10: weight["7.bias"], # conv2
                            16: weight["10.bias"],
                            18: weight["12.bias"],
                            20: weight["14.bias"], # conv3
                            26: weight["17.bias"],
                            28: weight["19.bias"],
                            30: weight["21.bias"], # conv4
                            36: weight["24.bias"],
                            38: weight["26.bias"],
                            40: weight["28.bias"], # conv5
                            43: weight["31.bias"], # conv6
                            45: weight["33.bias"], # conv7
                        }
                }
    elif config.vgg_config == [64, 64, "F11", "F12", 128, 128, "F21", "F22", 256, 256, 256, "F31", "F32",
                               512, 512, 512, "F41", "F42", 512, 512, 512]:
        print("load weight for fracPool w/o conv+ReLU.")
        weight_dict = \
                {"weight":
                        {
                            0: weight["0.weight"],
                            2: weight["2.weight"], # conv1
                            6: weight["5.weight"],
                            8: weight["7.weight"], # conv2
                            12: weight["10.weight"],
                            14: weight["12.weight"],
                            16: weight["14.weight"], # conv3
                            20: weight["17.weight"],
                            22: weight["19.weight"],
                            24: weight["21.weight"], # conv4
                            28: weight["24.weight"],
                            30: weight["26.weight"],
                            32: weight["28.weight"], # conv5
                            35: weight["31.weight"], # conv6
                            37: weight["33.weight"], # conv7
                        },
                "bias":
                        {
                            0: weight["0.bias"],
                            2: weight["2.bias"], # conv1
                            6: weight["5.bias"],
                            8: weight["7.bias"], # conv2
                            12: weight["10.bias"],
                            14: weight["12.bias"],
                            16: weight["14.bias"], # conv3
                            20: weight["17.bias"],
                            22: weight["19.bias"],
                            24: weight["21.bias"], # conv4
                            28: weight["24.bias"],
                            30: weight["26.bias"],
                            32: weight["28.bias"], # conv5
                            35: weight["31.bias"], # conv6
                            37: weight["33.bias"], # conv7
                        }
                }
        
    return weight_dict