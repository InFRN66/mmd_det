import torch
import torch.nn as nn
from torch.nn import Conv2d, ModuleList, Sequential, ReLU
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
import numpy as np
from ..utils import box_utils

from typing import List, Tuple
from ..nn.resnet import select_resnet
from .predictor import Predictor
# from .config import resnet50_ssd_config as config

from collections import OrderedDict

def create_resnet50_ssd(num_classes, config, is_test=False):
    num_anchors = config.num_anchors
    base_net = select_resnet('resnet50')()
    extras = ModuleList([
        # conv8
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv9
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv10
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            ReLU()
        ),
        # conv11
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            ReLU()
        )
    ])


    # for anchors 466644
    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * 4, kernel_size=3, padding=1),
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * num_classes, kernel_size=3, padding=1),
    ])

    # # for 666644 anchors
    # regression_headers = ModuleList([
    #     Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
    # ])

    # classification_headers = ModuleList([
    #     Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
    # ])

    return Res_SSD(base_net, extras, classification_headers, regression_headers, is_test, config, num_classes)


def create_resnet50_ssd_v2(num_classes, config, is_test=False):
    num_anchors = config.num_anchors
    resnet_clf = models.resnet50(pretrained=True)
    base_net = ModuleList([ x for i, x in enumerate(resnet_clf.children()) if i<7 ]) # until layer3
    extras = ModuleList([
        # conv8
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv9
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        # conv10
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            ReLU()
        ),
        # conv11
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            ReLU()
        )
    ])

    # for anchors 466644
    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * 4, kernel_size=3, padding=1),
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=num_anchors[0] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=num_anchors[1] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=num_anchors[2] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[3] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[4] * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=num_anchors[5] * num_classes, kernel_size=3, padding=1),
    ])

    # # for 666644 anchors
    # regression_headers = ModuleList([
    #     Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
    # ])

    # classification_headers = ModuleList([
    #     Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
    #     Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
    # ])

    return Res_SSD_v2(base_net, extras, classification_headers, regression_headers, is_test, config, num_classes)


def create_resnet50_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None, config=None):
    predictor = Predictor(net, config.image_size, mean=config.image_mean, std=config.image_std, 
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # init.constant(self.weight,self.gamma)
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class Res_SSD(nn.Module):
    def __init__(self, base_net, extras, classification_headers, regression_headers, is_test=False,
                 config=None, num_classes=21, device=None):
        super(Res_SSD, self).__init__()

        feature_layer = [[10, 16, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]]

        self.num_classes = num_classes
        # SSD network
        self.base_net = nn.ModuleList(base_net)
        self.norm = L2Norm(feature_layer[1][0], 20) # n_channels, scale
        self.extras = nn.ModuleList(extras)
        self.regression_headers = nn.ModuleList(regression_headers)
        self.classification_headers = nn.ModuleList(classification_headers)

        self.is_test = is_test
        self.config = config
        # self.feature_layer = feature_layer[0]
        self.feature_layer = [10, 16]

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []

        header_index = 0
        for i, layer in enumerate(self.base_net):
            x = layer(x)

            if i in [10,16]:
                if i == 10:
                    x = self.norm(x)
                
                confidence, location = self.compute_header(header_index, x)
                header_index += 1
                confidences.append(confidence)
                locations.append(location)

        for layer in self.extras:
            x = layer(x) # per sequential
            confidence, location = self.compute_header(header_index, x) 
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        # print('last', header_index) = 6
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            row_scores = confidences.clone() # softmax前のスコア
            confidences = F.softmax(confidences, dim=2)
            print(locations.shape, confidences.shape)
            return confidences, locations, row_scores

            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, row_scores 
        
        else:
            return confidences, locations # [batch, 8732, 21], [batch, 8732, 4]


    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location


    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)


    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)



# --- torchvision.models.resnet50 ---
class Res_SSD_v2(nn.Module):
    def __init__(self, base_net, extras, classification_headers, regression_headers, is_test=False,
                 config=None, num_classes=21, device=None):
        super(Res_SSD_v2, self).__init__()

        feature_layer = [[10, 16, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]]

        self.num_classes = num_classes
        self.base_net = nn.ModuleList(base_net)
        # self.source_layer_add_ons = ModuleList([
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # ])
        self.source_layer_add_ons = ModuleList([
            L2Norm(feature_layer[1][0], 20) # n_channels, scale
        ])
        # self.norm = L2Norm(feature_layer[1][0], 20) # n_channels, scale
        self.norm = self.source_layer_add_ons[0] # --- miss?
        self.extras = nn.ModuleList(extras)
        self.regression_headers = nn.ModuleList(regression_headers)
        self.classification_headers = nn.ModuleList(classification_headers)

        self.is_test = is_test
        self.config = config
        self.feature_layer = [10, 16]

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []

        header_index = 0
        # --- conv1 -> resnet.layer3
        for i, layer in enumerate(self.base_net): # 1.[0:]6
            x = layer(x)

            if i in [5,6]:
                if i == 5:
                    x = self.norm(x)
                
                confidence, location = self.compute_header(header_index, x)
                header_index += 1
                confidences.append(confidence)
                locations.append(location)

        for layer in self.extras:
            x = layer(x) # per sequential
            confidence, location = self.compute_header(header_index, x) 
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        # print('last', header_index) = 6
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        

        if self.is_test:
            row_scores = confidences.clone() # softmax前のスコア
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, row_scores 
        
        else:
            # print(confidences.shape)
            assert (confidences.shape[1] == 8732) and (locations.shape[1] == 8732)
            return confidences, locations # [batch, 8732, 21], [batch, 8732, 4]


    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location


    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)


    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

    
def modifie_weight(net, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    net_dict = net.state_dict() # base_net, num_batch...
    pretrained_dict = convert_key(pretrained_dict)

    # # 1. fliter \ netにないkeyは消す
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    # # 2. overwrite
    net_dict.update(pretrained_dict)
    # # 3. load
    net.load_state_dict(net_dict)
    
# --- 4,6,6,6,4,4でも適用したい時用  
# def convert_key(pretrained_dict):
#     new_pretrained_dict = OrderedDict()
#     extras_map = {"0": "0.0", "1": "0.1", "2": "1.0", "3": "1.1", "4": "2.0",  "5": "2.1", "6": "3.0", "7": "3.1"}
#     for i in pretrained_dict.keys():
#         # base -> base_net
#         if i.startswith("base"):
#             key = i.replace("base", "base_net")
#             new_pretrained_dict[key] = pretrained_dict[i]
#         # loc -> regression_headers
#         elif i.startswith("loc"):
#             if (i.startswith("loc.0")):
#                 continue
#             else:
#                 key = i.replace("loc", "regression_headers")
#                 new_pretrained_dict[key] = pretrained_dict[i]
#         # conf -> classification_headers
#         elif i.startswith("conf"):
#             if (i.startswith("conf.0")):
#                 continue
#             else:
#                 key = i.replace("conf", "classification_headers")
#                 new_pretrained_dict[key] = pretrained_dict[i]
#         # e.g. 0 -> 0.0 / 1 -> 0.1 ...
#         elif i.startswith("extras"):
#             key = i.replace(i.split(".")[1], extras_map[i.split(".")[1]])
#             new_pretrained_dict[key] = pretrained_dict[i]
#         else:
#             new_pretrained_dict[i] = pretrained_dict[i]
#     return new_pretrained_dict


def convert_key(pretrained_dict):
    new_pretrained_dict = OrderedDict()
    extras_map = {"0": "0.0", "1": "0.1", "2": "1.0", "3": "1.1", "4": "2.0",  "5": "2.1", "6": "3.0", "7": "3.1"}
    for i in pretrained_dict.keys():
        # base -> base_net
        if i.startswith("base"):
            key = i.replace("base", "base_net")
            new_pretrained_dict[key] = pretrained_dict[i]
        # loc -> regression_headers
        elif i.startswith("loc"):
            key = i.replace("loc", "regression_headers")
            new_pretrained_dict[key] = pretrained_dict[i]
        # conf -> classification_headers
        elif i.startswith("conf"):
            key = i.replace("conf", "classification_headers")
            new_pretrained_dict[key] = pretrained_dict[i]
        # e.g. 0 -> 0.0 / 1 -> 0.1 ...
        elif i.startswith("extras"):
            key = i.replace(i.split(".")[1], extras_map[i.split(".")[1]])
            new_pretrained_dict[key] = pretrained_dict[i]
        else:
            new_pretrained_dict[i] = pretrained_dict[i]
    return new_pretrained_dict




# resnet50のアンカー(all6. 466644)を試すためにweightの入れ替え
"""
my_resnet.load_state_dict(torch.load("../models/resnet50_base.pth"))

my_resnet.base_net[0].weight = RES.conv1.weight
my_resnet.base_net[1].weight = RES.bn1.weight

minas = 4
for i in range(4,7):
    my_resnet.base_net[i].conv1.weight = RES.layer1[i-minas].conv1.weight
    my_resnet.base_net[i].bn1.weight = RES.layer1[i-minas].bn1.weight
    my_resnet.base_net[i].conv2.weight = RES.layer1[i-minas].conv2.weight
    my_resnet.base_net[i].bn2.weight = RES.layer1[i-minas].bn2.weight
    my_resnet.base_net[i].conv3.weight = RES.layer1[i-minas].conv3.weight
    my_resnet.base_net[i].bn3.weight = RES.layer1[i-minas].bn3.weight
    try:
        my_resnet.base_net[i].downsample[0].weight = RES.layer1[i-minas].downsample[0].weight
        my_resnet.base_net[i].downsample[1].weight = RES.layer1[i-minas].downsample[1].weight
        print('Seq11')
    except:
        pass


minas = 7
for i in range(7,11):
    my_resnet.base_net[i].conv1.weight = RES.layer2[i-minas].conv1.weight
    my_resnet.base_net[i].bn1.weight = RES.layer2[i-minas].bn1.weight
    my_resnet.base_net[i].conv2.weight = RES.layer2[i-minas].conv2.weight
    my_resnet.base_net[i].bn2.weight = RES.layer2[i-minas].bn2.weight
    my_resnet.base_net[i].conv3.weight = RES.layer2[i-minas].conv3.weight
    my_resnet.base_net[i].bn3.weight = RES.layer2[i-minas].bn3.weight
    try:
        
        my_resnet.base_net[i].downsample[0].weight = RES.layer2[i-minas].downsample[0].weight
        my_resnet.base_net[i].downsample[1].weight = RES.layer2[i-minas].downsample[1].weight
        print('Seq22')
    except:
        pass
    
minas = 11
for i in range(11, 16):
    my_resnet.base_net[i].conv1.weight = RES.layer3[i-minas].conv1.weight
    my_resnet.base_net[i].bn1.weight = RES.layer3[i-minas].bn1.weight
    my_resnet.base_net[i].conv2.weight = RES.layer3[i-minas].conv2.weight
    my_resnet.base_net[i].bn2.weight = RES.layer3[i-minas].bn2.weight
    my_resnet.base_net[i].conv3.weight = RES.layer3[i-minas].conv3.weight
    my_resnet.base_net[i].bn3.weight = RES.layer3[i-minas].bn3.weight
    try:
        my_resnet.base_net[i].downsample[0].weight = RES.layer3[i-minas].downsample[0].weight
        my_resnet.base_net[i].downsample[1].weight = RES.layer3[i-minas].downsample[1].weight
        print('Seq33')
    except:
        pass
"""

    # def forward_resnet(self, x, phase='eval'):
    #     """Applies network layers and ops on input image(s) x.

    #     Args:
    #         x: input image or batch of images. Shape: [batch,3,300,300].

    #     Return:
    #         Depending on phase:
    #         test:
    #             Variable(tensor) of output class label predictions,
    #             confidence score, and corresponding location predictions for
    #             each object detected. Shape: [batch,topk,7]

    #         train:
    #             list of concat outputs from:
    #                 1: confidence layers, Shape: [batch*num_priors,num_classes]
    #                 2: localization layers, Shape: [batch,num_priors*4]

    #         feature:
    #             the features maps of the feature extractor
    #     """
    #     sources, loc, conf = [list() for _ in range(3)]

    #     # apply bases layers and cache source layer outputs
    #     for k in range(len(self.base_net)):
    #         x = self.base_net[k](x)
    #         if k in self.feature_layer: # if in [10, 16, 'S', 'S', '', '']
    #             if len(sources) == 0: # [10]
    #                 s = self.norm(x) # L2Norm[10]
    #                 sources.append(s)
    #             else:
    #                 sources.append(x)

    #     # apply extra layers and cache source layer outputs
    #     for k, v in enumerate(self.extras):
    #         # TODO:maybe donot needs the relu here
    #         x = F.relu(v(x), inplace=True)
    #         # TODO:lite is different in here, should be changed
    #         if k % 2 == 1:
    #             sources.append(x)
        
    #     if phase == 'feature':
    #         return sources

    #     # apply multibox head to source layers
    #     for (x, l, c) in zip(sources, self.loc, self.conf):
    #         loc.append(l(x).permute(0, 2, 3, 1).contiguous())
    #         conf.append(c(x).permute(0, 2, 3, 1).contiguous())
    #     loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    #     conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    #     if phase == 'eval':
    #         output = (
    #             loc.view(loc.size(0), -1, 4),                   # loc preds
    #             self.softmax(conf.view(-1, self.num_classes)),  # conf preds
    #             conf.view(-1, self.num_classes)
    #         )
    #     else:
    #         output = (
    #             loc.view(loc.size(0), -1, 4),
    #             conf.view(conf.size(0), -1, self.num_classes),
    #         )
    #     return output
