import torch
import torch.nn as nn
import os
import sys
import math
import numpy as np
from copy import deepcopy


from .darknet53 import DarkNet53, DarkNetBlock, Conv_Bn_Leaky
from .utils import (convert_scores_to_sigmoid, convert_coordinates_to_offset,
                    targets_match_all, permute_sigmoid, decode)


class DetectionLayer(nn.Module):
    def __init__(self, cfg, is_test):
        super(DetectionLayer, self).__init__()
        self.input_size = cfg.image_size
        self.num_classes = cfg.num_classes
        self.all_anchors = cfg.all_anchors
        self.grids = cfg.grids
        self.is_test = is_test

    def forward(self, detections):
        # detections = convert_output_to_offset(detections, self.input_wh, self.num_classes, self.all_anchors)
        detections = convert_scores_to_sigmoid(detections, self.num_classes)
        sigmoid_prediction = [detections[i].clone() for i in range(len(detections))]
        sigmoid_offset_prediction = convert_coordinates_to_offset(detections, self.input_size, self.num_classes, self.grids, self.all_anchors)
        return torch.cat(sigmoid_prediction, 1), torch.cat(sigmoid_offset_prediction, 1)


class YOLOv3(nn.Module):
    def __init__(self, backbone, pred11, pred12, pred21, pred22, pred31, cfg, is_test):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.pred11 = pred11
        self.pred12 = pred12
        self.pred21 = pred21
        self.pred22 = pred22
        self.pred31 = pred31
        self.is_test = is_test
        self.detection_layer = DetectionLayer(cfg, self.is_test)


    def forward(self, x):
        """
        x: image
        """
        for_detection = list()
        C3, C4, C5 = self.backbone(x)

        out = C5
        for i in range(5):
            out = self.pred11[i](out)
        for_yolo = out
        for i in range(5, 7):
            for_yolo = self.pred11[i](for_yolo)
        for_detection.append(for_yolo)

        out = torch.cat((self.pred12(out), C4), 1)
        for i in range(5):
            out = self.pred21[i](out)
        for_yolo = out
        for i in range(5, 7):
            for_yolo = self.pred21[i](for_yolo)
        for_detection.append(for_yolo)

        out = torch.cat((self.pred22(out), C3), 1)
        for i in range(len(self.pred31)):
            out = self.pred31[i](out)
        for_detection.append(out)

        # if self.is_test:
        #     return self.detection_layer(for_detection)
        # else:
        #     return for_detection
        return self.detection_layer(for_detection)

def create_net(cfg, is_test=False):
    backbone = DarkNet53(cfg.blocks)

    prediction_11 = nn.ModuleList([
        Conv_Bn_Leaky(1024, 512, 1, 1, 0),
        Conv_Bn_Leaky(512, 1024, 3, 1, 1),
        Conv_Bn_Leaky(1024, 512, 1, 1, 0),
        Conv_Bn_Leaky(512, 1024, 3, 1, 1),
        Conv_Bn_Leaky(1024, 512, 1, 1, 0),
        Conv_Bn_Leaky(512, 1024, 3, 1, 1),
        nn.Conv2d(1024, 3*(5+cfg.num_classes), 1, 1, 0)
    ])
    prediction_12 = nn.Sequential(
        Conv_Bn_Leaky(512, 256, 1, 1, 0),
        nn.Upsample(scale_factor=2, mode="nearest")
    ) 

    prediction_21 = nn.ModuleList([
        Conv_Bn_Leaky((256+512), 256, 1, 1, 0),
        Conv_Bn_Leaky(256, 512, 3, 1, 1),
        Conv_Bn_Leaky(512, 256, 1, 1, 0),
        Conv_Bn_Leaky(256, 512, 3, 1, 1),
        Conv_Bn_Leaky(512, 256, 1, 1, 0),
        Conv_Bn_Leaky(256, 512, 3, 1, 1),
        nn.Conv2d(512, 3*(5+cfg.num_classes), 1, 1, 0)
    ])
    prediction_22 = nn.Sequential(
        Conv_Bn_Leaky(256, 128, 1, 1, 0),
        nn.Upsample(scale_factor=2, mode="nearest")
    )

    prediction_31 = nn.ModuleList([
        Conv_Bn_Leaky((128+256), 128, 1, 1, 0),
        Conv_Bn_Leaky(128, 256, 3, 1, 1),
        Conv_Bn_Leaky(256, 128, 1, 1, 0),
        Conv_Bn_Leaky(128, 256, 3, 1, 1),
        Conv_Bn_Leaky(256, 128, 1, 1, 0),
        Conv_Bn_Leaky(128, 256, 3, 1, 1),
        nn.Conv2d(256, 3*(5+cfg.num_classes), 1, 1, 0)
    ])
    return YOLOv3(backbone, prediction_11, prediction_12, prediction_21, prediction_22, prediction_31, cfg, is_test)


class WeightMseLoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightMseLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        ''' inputs is N * C
            targets is N * C
            weights is N * C
        '''
        N = inputs.size(0)
        C = inputs.size(1)

        out = (targets - inputs)
        out = weights * torch.pow(out, 2)
        loss = out.sum()

        if self.size_average:
            loss = loss / (N * C)
        return loss



class MultiboxLoss(nn.Module):
    def __init__(self, cfg, use_gpu=True):
        super(MultiboxLoss, self).__init__()
        self.num_classes = cfg.num_classes
        self.pos_iou_threshold = cfg.pos_iou_threshold
        self.use_gpu = use_gpu
        self.all_anchors = cfg.all_anchors
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.weight_mseloss = WeightMseLoss(size_average=False)
        self.input_wh = cfg.input_wh
        self.grids = cfg.grids
        self.anchor_list = list()
        for key in self.all_anchors:
            self.anchor_list += self.all_anchors[key] # [[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]]
        

    # def forward(self, detections, targets, debug=False):
    #     l_data, m_data, h_data = detections

    #     l_grid_wh = (l_data.size(3), l_data.size(2))
    #     m_grid_wh = (m_data.size(3), m_data.size(2))
    #     h_grid_wh = (h_data.size(3), h_data.size(2))
    #     feature_dim = (l_grid_wh, m_grid_wh, h_grid_wh)
    #     batch_size = l_data.size(0)
    #     pred_l, stride_l = permute_sigmoid(l_data, self.input_wh, 3, self.num_classes)
    #     pred_m, stride_m = permute_sigmoid(m_data, self.input_wh, 3, self.num_classes)
    #     pred_h, stride_h = permute_sigmoid(h_data, self.input_wh, 3, self.num_classes)

    #     print("stride: ", stride_l, stride_m, stride_h)
    #     pred = torch.cat((pred_l, pred_m, pred_h), 1) # (batch, num_anchors, 85) before decode (pure output, not offset)
    #     print("pred", pred.shape, pred[0, :3, :])
    #     # anchors1 = self.anchors[self.anchors_mask[0][0]: self.anchors_mask[0][-1]+1]
    #     # anchors2 = self.anchors[self.anchors_mask[1][0]: self.anchors_mask[1][-1]+1]
    #     # anchors3 = self.anchors[self.anchors_mask[2][0]: self.anchors_mask[2][-1]+1]
    #     anchors1 = [[116, 90], [156, 198], [373, 326]]
    #     anchors2 = [[30, 61], [62, 45], [59, 119]]
    #     anchors3 = [[10, 13], [16, 30], [33, 23]]

    #     decode_l = decode(pred_l.new_tensor(pred_l).detach(), self.input_wh, anchors1, self.num_classes, stride_l)
    #     print("decode_l", decode_l.shape, decode_l[0, :3, :])

    #     decode_m = decode(pred_m.new_tensor(pred_m).detach(), self.input_wh, anchors2, self.num_classes, stride_m)
    #     decode_h = decode(pred_h.new_tensor(pred_h).detach(), self.input_wh, anchors3, self.num_classes, stride_h)
    #     decode_pred = torch.cat((decode_l, decode_m, decode_h), 1)
    #     # ::: preprocess. same as prediction and train

    #     # detections: [batch, (large+middle+small), (5+classes)]
    #     # batch_size, num_pred = detections.shape[:2]
    #     num_pred = pred_l.size(1) + pred_m.size(1) + pred_h.size(1)


    #     # prediction targets x,y,w,h,objectness, class
    #     pred_t = torch.Tensor(batch_size, num_pred, 6).cuda()
    #     # xywh scale, scale = 2 - truth.w * truth.h (if truth is normlized to 1)
    #     scale_t = torch.FloatTensor(batch_size, num_pred).cuda()
    #     # foreground targets mask
    #     fore_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

    #     # background targets mask, we only calculate the objectness pred loss 
    #     back_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

    #     print("1:fore_mask_t", fore_mask_t)
    #     for idx in range(batch_size):
    #         # match all targets
    #         targets_match_all(self.input_wh, self.pos_iou_threshold, targets[idx], decode_pred[idx][:, :4], self.anchor_list, self.grids,
    #                           pred_t, scale_t, fore_mask_t, back_mask_t, num_pred, idx)
    #     print("2:fore_mask_t", fore_mask_t)

    #     scale_factor = scale_t[fore_mask_t].view(-1, 1)
    #     scale_factor = scale_factor.expand((scale_factor.size(0), 2))
    #     cls_t = pred_t[..., 5][fore_mask_t].long().view(-1, 1)


    #     # print("detections: ", detections[0, :3, 5:])

    #     cls_pred = pred[..., 5:]

    #     # cls loss
    #     cls_fore_mask_t = fore_mask_t.new_tensor(fore_mask_t).view(batch_size, num_pred, 1).expand_as(cls_pred)
    #     cls_pred = cls_pred[cls_fore_mask_t].view(-1, self.num_classes)
    #     class_mask = cls_pred.data.new(cls_t.size(0), self.num_classes).fill_(0)
    #     class_mask.scatter_(1, cls_t, 1.)
    #     cls_loss = self.bce_loss(cls_pred, class_mask)
    #     ave_cls = (class_mask * cls_pred).sum().item() / cls_pred.size(0)
        
    #     # conf loss
    #     conf_t = pred_t[..., 4]
    #     fore_conf_t = conf_t[fore_mask_t].view(-1, 1)
    #     back_conf_t = conf_t[back_mask_t].view(-1, 1)
    #     fore_conf_pred = pred[..., 4][fore_mask_t].view(-1, 1)
    #     back_conf_pred = pred[..., 4][back_mask_t].view(-1, 1)
    #     fore_num = fore_conf_pred.size(0)
    #     back_num = back_conf_pred.size(0)
    #     Obj = fore_conf_pred.sum().item() / fore_num
    #     no_obj = back_conf_pred.sum().item() / back_num

    #     fore_conf_loss = self.bce_loss(fore_conf_pred, fore_conf_t)
    #     back_conf_loss = self.bce_loss(back_conf_pred, back_conf_t)
    #     conf_loss = fore_conf_loss + back_conf_loss  

    #     # loc loss
    #     loc_pred = pred[..., :4]
    #     loc_t = pred_t[..., :4]
    #     fore_mask_t = fore_mask_t.view(batch_size, num_pred, 1).expand_as(loc_pred)
    #     loc_t = loc_t[fore_mask_t].view(-1, 4)
    #     loc_pred = loc_pred[fore_mask_t].view(-1, 4)

    #     xy_t, wh_t = loc_t[:, :2], loc_t[:, 2:]
    #     xy_pred, wh_pred = loc_pred[:, :2], loc_pred[:, 2:]
    #     # xy_loss = F.binary_cross_entropy(xy_pred, xy_t, scale_factor, size_average=False)

    #     xy_loss = self.weight_mseloss(xy_pred, xy_t, scale_factor) / 2
    #     wh_loss = self.weight_mseloss(wh_pred, wh_t, scale_factor) / 2

    #     loc_loss = xy_loss + wh_loss        

    #     loc_loss /= batch_size
    #     conf_loss /= batch_size
    #     cls_loss /= batch_size

    #     if debug:
    #         print("xy_loss", round(xy_loss.item(), 5), "wh_loss", round(wh_loss.item(), 5), "cls_loss", round(cls_loss.item(), 5), "ave_cls", round(ave_cls, 5), "Obj", round(Obj, 5), "no_obj", round(no_obj, 5), "fore_conf_loss", round(fore_conf_loss.item(), 5),
    #             "back_conf_loss", round(back_conf_loss.item(), 5))

    #     loss = loc_loss + conf_loss + cls_loss

    #     return loss


    def forward(self, add_sigmoid,  add_sigmoid_offset, targets, debug=False):
        """
        l_data, m_data, h_data = detections

        l_grid_wh = (l_data.size(3), l_data.size(2))
        m_grid_wh = (m_data.size(3), m_data.size(2))
        h_grid_wh = (h_data.size(3), h_data.size(2))
        feature_dim = (l_grid_wh, m_grid_wh, h_grid_wh)
        batch_size = l_data.size(0)
        pred_l, stride_l = permute_sigmoid(l_data, self.input_wh, 3, self.num_classes)
        pred_m, stride_m = permute_sigmoid(m_data, self.input_wh, 3, self.num_classes)
        pred_h, stride_h = permute_sigmoid(h_data, self.input_wh, 3, self.num_classes)

        print("stride: ", stride_l, stride_m, stride_h)
        pred = torch.cat((pred_l, pred_m, pred_h), 1) # (batch, num_anchors, 85) before decode (pure output, not offset)
        print("pred", pred.shape, pred[0, :3, :])
        # anchors1 = self.anchors[self.anchors_mask[0][0]: self.anchors_mask[0][-1]+1]
        # anchors2 = self.anchors[self.anchors_mask[1][0]: self.anchors_mask[1][-1]+1]
        # anchors3 = self.anchors[self.anchors_mask[2][0]: self.anchors_mask[2][-1]+1]
        anchors1 = [[116, 90], [156, 198], [373, 326]]
        anchors2 = [[30, 61], [62, 45], [59, 119]]
        anchors3 = [[10, 13], [16, 30], [33, 23]]

        decode_l = decode(pred_l.new_tensor(pred_l).detach(), self.input_wh, anchors1, self.num_classes, stride_l)
        print("decode_l", decode_l.shape, decode_l[0, :3, :])

        decode_m = decode(pred_m.new_tensor(pred_m).detach(), self.input_wh, anchors2, self.num_classes, stride_m)
        decode_h = decode(pred_h.new_tensor(pred_h).detach(), self.input_wh, anchors3, self.num_classes, stride_h)
        decode_pred = torch.cat((decode_l, decode_m, decode_h), 1)
        # ::: preprocess. same as prediction and train
        """

        # detections: [batch, (large+middle+small), (5+classes)]
        batch_size, num_pred = add_sigmoid_offset.shape[:2]

        # prediction targets x,y,w,h,objectness, class
        pred_t = torch.Tensor(batch_size, num_pred, 6).cuda()
        # xywh scale, scale = 2 - truth.w * truth.h (if truth is normlized to 1)
        scale_t = torch.FloatTensor(batch_size, num_pred).cuda()
        # foreground targets mask
        fore_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

        # background targets mask, we only calculate the objectness pred loss 
        back_mask_t = torch.ByteTensor(batch_size, num_pred).cuda()

        print("1:fore_mask_t", fore_mask_t)
        for idx in range(batch_size):
            # match all targets
            targets_match_all(self.input_wh, self.pos_iou_threshold, targets[idx], add_sigmoid_offset[idx][:, :4], self.anchor_list, self.grids,
                              pred_t, scale_t, fore_mask_t, back_mask_t, num_pred, idx)
        print("2:fore_mask_t", fore_mask_t)

        scale_factor = scale_t[fore_mask_t].view(-1, 1)
        scale_factor = scale_factor.expand((scale_factor.size(0), 2))
        cls_t = pred_t[..., 5][fore_mask_t].long().view(-1, 1)

        cls_pred = add_sigmoid[..., 5:]

        # cls loss
        cls_fore_mask_t = fore_mask_t.new_tensor(fore_mask_t).view(batch_size, num_pred, 1).expand_as(cls_pred)
        cls_pred = cls_pred[cls_fore_mask_t].view(-1, self.num_classes)
        class_mask = cls_pred.data.new(cls_t.size(0), self.num_classes).fill_(0)
        class_mask.scatter_(1, cls_t, 1.)
        cls_loss = self.bce_loss(cls_pred, class_mask)
        ave_cls = (class_mask * cls_pred).sum().item() / cls_pred.size(0)
        
        # conf loss
        conf_t = pred_t[..., 4]
        fore_conf_t = conf_t[fore_mask_t].view(-1, 1)
        back_conf_t = conf_t[back_mask_t].view(-1, 1)
        fore_conf_pred = add_sigmoid[..., 4][fore_mask_t].view(-1, 1)
        back_conf_pred = add_sigmoid[..., 4][back_mask_t].view(-1, 1)
        fore_num = fore_conf_pred.size(0)
        back_num = back_conf_pred.size(0)
        Obj = fore_conf_pred.sum().item() / fore_num
        no_obj = back_conf_pred.sum().item() / back_num

        fore_conf_loss = self.bce_loss(fore_conf_pred, fore_conf_t)
        back_conf_loss = self.bce_loss(back_conf_pred, back_conf_t)
        conf_loss = fore_conf_loss + back_conf_loss  

        # loc loss
        loc_pred = add_sigmoid[..., :4]
        loc_t = pred_t[..., :4]
        fore_mask_t = fore_mask_t.view(batch_size, num_pred, 1).expand_as(loc_pred)
        loc_t = loc_t[fore_mask_t].view(-1, 4)
        loc_pred = loc_pred[fore_mask_t].view(-1, 4)

        xy_t, wh_t = loc_t[:, :2], loc_t[:, 2:]
        xy_pred, wh_pred = loc_pred[:, :2], loc_pred[:, 2:]
        # xy_loss = F.binary_cross_entropy(xy_pred, xy_t, scale_factor, size_average=False)

        xy_loss = self.weight_mseloss(xy_pred, xy_t, scale_factor) / 2
        wh_loss = self.weight_mseloss(wh_pred, wh_t, scale_factor) / 2

        loc_loss = xy_loss + wh_loss        

        loc_loss /= batch_size
        conf_loss /= batch_size
        cls_loss /= batch_size

        if debug:
            print("xy_loss", round(xy_loss.item(), 5), "wh_loss", round(wh_loss.item(), 5), "cls_loss", round(cls_loss.item(), 5), "ave_cls", round(ave_cls, 5), "Obj", round(Obj, 5), "no_obj", round(no_obj, 5), "fore_conf_loss", round(fore_conf_loss.item(), 5),
                "back_conf_loss", round(back_conf_loss.item(), 5))

        loss = loc_loss + conf_loss + cls_loss

        return loss