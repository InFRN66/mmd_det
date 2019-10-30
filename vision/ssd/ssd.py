import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils, box_utils_copy
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """
        Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes # source_layer_indexes = [ (23, BatchNorm2d(512)), len(base_net), ]
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        # ModuleList((0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)]) # batchNorm2D
        """
        self.source_layer_add_ons == 
        ModuleList(
            (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        """
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
        start_layer_index = 0
        header_index = 0
        """
        self.source_layer_indexes == 
            [(23, BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), 
             35]
            /tuple /int

            [(23, BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), 
             24, 
             36]
        """
        # print("@ssd: self.source_layer_indexes", self.source_layer_indexes)
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple): # (23, BatchNorm2d(512))
                added_layer = end_layer_index[1] # BatchNorm
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None

            # first : (1,3,300,300) - (1,512,38,38) Conv4-3 (0 - 23)
            # second: (1,512,38,38) - (1,1024,19,19) Conv7
            # # --- for Fractional
            # if end_layer_index == len(self.base_net):
            #     for layer in self.base_net[start_layer_index: end_layer_index]: # 1.[0:23] 2.[23:35]
            #         additional_x = layer(x)
            #     start_layer_index = end_layer_index
            #     confidence, location = self.compute_header(header_index, additional_x)
            #     header_index += 1
            #     # print("in for", confidence.shape, header_index, y.shape)
            #     confidences.append(confidence)
            #     locations.append(location)
            #     continue
            # else:
            #     for layer in self.base_net[start_layer_index: end_layer_index]: # 1.[0:23] 2.[23:35]
            #         x = layer(x)
            # # --- for Fractional

            for layer in self.base_net[start_layer_index: end_layer_index]: # 1.[0:23] 2.[23:35]
                x = layer(x)

            if added_layer:
                y = added_layer(x)
            else:
                y = x

            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1

            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1) # concat[1, 5776, 21] + [1,7942,21] + ...
        locations = torch.cat(locations, 1)     # concat[1, 5776, 4] + [1, 7942, 4] + ...

        if self.is_test:
            row_scores = confidences.clone() # softmax前のスコア
            confidences = F.softmax(confidences, dim=2)
            # --- 
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, row_scores
                        
        else:
            return confidences, locations # [batch, 8732, 21], [batch, 8732, 4]


    def compute_header(self, i, x):
        # print("compute_head",i,x.shape)
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


class SSD_complement_with_fractional(SSD):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, complement_fractional: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """
        Compose a SSD model using the given components.
        """
        super(SSD_complement_with_fractional, self).__init__(
            num_classes, base_net, source_layer_indexes, extras, classification_headers,
            regression_headers, is_test, config, device)
        print("similar class in SSD")
        self.complement_fractional = complement_fractional
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        fractional_index = 0
        """
        self.source_layer_indexes == 
            [(23, BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), 
             35]
            /tuple /int

            [(23, BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), 
             24, 
             36]
        """
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple): # (23, BatchNorm2d(512))
                added_layer = end_layer_index[1] # BatchNorm
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None

            for layer in self.base_net[start_layer_index: end_layer_index]: # 1.[0:23] 2.[23:35]
                x = layer(x)

            if added_layer:
                y = added_layer(x)
            else:
                y = x

            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1

            start_layer_index = end_layer_index
            # normal layer
            # print("y", y.shape)
            confidence, location = self.compute_header(header_index, y)
            confidences.append(confidence)
            locations.append(location)
            header_index += 1
            # fractional pooling
            if fractional_index != len(self.complement_fractional):    
                confidence, location = self.compute_fractional(header_index, fractional_index, y)
                confidences.append(confidence)
                locations.append(location)
                header_index += 1
                fractional_index += 1
            

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            confidences.append(confidence)
            locations.append(location)
            header_index += 1
            if fractional_index != len(self.complement_fractional):    
                confidence, location = self.compute_fractional(header_index, fractional_index, x)
                confidences.append(confidence)
                locations.append(location)
                header_index += 1
                fractional_index += 1

        confidences = torch.cat(confidences, 1) # concat[1, 5776, 21] + [1,7942,21] + ...
        locations = torch.cat(locations, 1)     # concat[1, 5776, 4] + [1, 7942, 4] + ...

        if self.is_test:
            row_scores = confidences.clone() # softmax前のスコア
            confidences = F.softmax(confidences, dim=2)
            # --- 
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes, row_scores
                        
        else:
            return confidences, locations # [batch, 8732, 21], [batch, 8732, 4]

    def compute_fractional(self, i, fi, x):
        # print("comute_fractional", i, fi, x.shape)
        x = self.complement_fractional[fi](x)
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location



class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold, path=None, stat=False, sigmoid_thresh=None):
    # def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold, **kwargs):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.stat = stat
        self.path = path
        self.sigmoid_thresh = sigmoid_thresh

    def __call__(self, gt_boxes, gt_labels):
        """
        gt_boxes: normalized scale [num_targets, 4]
        gt_labels: labels per objects [num_targets]

        self.corner_form_priors: [8732, 4] / x1y1x2y2 in 01 scale
        self.center_form_priors: [8732, 4] / cxcywh

        boxes: 各priorが, 全targetのgt_boxes内で最も近い"gtのボックス座標"を保持
        [prior0:target1の[x1, y1, x2, y2], prior1:target1の[], prior2:target0の[] ...], cornerform
        labels: 各priorが, 全targetのgt_labels内で最も近いボックスのlabel情報を保持
        [prior0[target1のクラス], prior1[target1のクラス], prior2[target0のクラス] ...]
        """
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        # --- add for count anchors ---
        if self.stat: # count positive sample in each gt
            boxes, labels, count_boxes, count_labels = box_utils.assign_priors_for_count(gt_boxes, gt_labels,
                                                                                         self.corner_form_priors, self.iou_threshold)
            gt_size, each_anchor_indicies, each_anchor_num =\
                box_utils.aggregate_anchors(gt_boxes, gt_labels, count_boxes,
                                            count_labels,  self.corner_form_priors)
            box_utils.for_statics_anchors(
                gt_boxes, gt_labels, gt_size, each_anchor_indicies, each_anchor_num, path=self.path)
            return boxes, labels

        elif self.sigmoid_thresh:
            # print("Sigmoid sampling ...")
            boxes, labels, mask = box_utils.assign_priors_for_sigmoid_masking(gt_boxes, gt_labels,
                                                                              self.corner_form_priors,
                                                                              self.sigmoid_thresh["alpha"],
                                                                              self.sigmoid_thresh["beta"],
                                                                              self.sigmoid_thresh["alpha_y"]
                                                                              )
            boxes = box_utils.corner_form_to_center_form(boxes)
            locations = box_utils.convert_boxes_to_locations(boxes,
                                                             self.center_form_priors,
                                                             self.center_variance,
                                                             self.size_variance)
            return locations, labels, mask  # [8732, 4], [8732]

        else: # original
            boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                    self.corner_form_priors, self.iou_threshold)
            # --- need for original ---
            boxes = box_utils.corner_form_to_center_form(boxes) # [x1y1x2y2] -> [cxcywh]
            # --- 実際のgt_boxとpriors座標を，lossで使うための差分に変換(g^ = (g - d)/d, etc)
            locations = box_utils.convert_boxes_to_locations(boxes,
                                                             self.center_form_priors,
                                                             self.center_variance,
                                                             self.size_variance)
            return locations, labels  # [8732, 4], [8732]      


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def exchange_weight_for_resnet(my_net, DEF):
    '''
    Memo:
        my_net : resnet50 in ssd 
        DEF : default(torchvision.models) resnet50
    '''

    my_net.base_net[0].weight = DEF.conv1.weight
    my_net.base_net[0].bias = DEF.conv1.bias
    my_net.base_net[1].weight = DEF.bn1.weight
    my_net.base_net[1].bias = DEF.bn1.bias

    for i in range(4,7):
    
        for l1, l2 in zip(my_net.base_net[i].children(), DEF.layer1[i-4].children()):
            print(l1, l2, l1.__class__.__name__, l2.__class__.__name__)
            try:
                l1.weight = l2.weight
            except:
                if l1.__class__.__name__ == l2.__class__.__name__ == 'ReLU':
                    print('ReeL')
                    pass
                elif l1.__class__.__name__ == l2.__class__.__name__ == 'Sequential':
                    print('Sequential')
                    for ll1, ll2 in zip(l1.children(), l2.children()):
                        ll1.weight = ll2.weight
            print()
    print('==='*30)
    for i in range(7, 11):
        
        for l1, l2 in zip(my_net.base_net[i].children(), DEF.layer2[i-7].children()):
            print(l1, l2, l1.__class__.__name__, l2.__class__.__name__)
            try:
                l1.weight = l2.weight
            except:
                if l1.__class__.__name__ == l2.__class__.__name__ == 'ReLU':
                    print('ReeL')
                    pass
                elif l1.__class__.__name__ == l2.__class__.__name__ == 'Sequential':
                    print('Sequential')
                    for ll1, ll2 in zip(l1.children(), l2.children()):
                        ll1.weight = ll2.weight
            print()
            
    print('==='*30)
    for i in range(11, 17):
        
        for l1, l2 in zip(my_net.base_net[i].children(), DEF.layer3[i-11].children()):
            print(l1, l2, l1.__class__.__name__, l2.__class__.__name__)
            try:
                l1.weight = l2.weight
            except:
                if l1.__class__.__name__ == l2.__class__.__name__ == 'ReLU':
                    print('ReeL')
                    pass
                elif l1.__class__.__name__ == l2.__class__.__name__ == 'Sequential':
                    print('Sequential')
                    for ll1, ll2 in zip(l1.children(), l2.children()):
                        ll1.weight = ll2.weight
            print()

    # torch.save(my_net.state_dict(), 'models/default/resnet50-base-ReLU.pth')
    # other_net.load_state_dict(torch.load('models/default/resnet50-base-ReLU.pth'))
