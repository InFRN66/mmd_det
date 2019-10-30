import numpy as np
import math
import torch
import torch.nn as nn

# def convert_output_to_offset(detections, input_wh, num_classes, all_anchors):
#     """
#     detections: list, [batch, amchors*(5+classes), grid, grid]
#     input_wh: input image size
#     num_classes: classes to detect
#     all_anchors: list of all anchor boxes e.g., [[w1,h1], [w2,h2], [w3,h3], ...] 
#     """
#     new_detections = list()
#     for i in range(len(detections)):
#         det = detections[i]
#         batch = det.shape[0]
#         grid_size = det.shape[-1] # w or h
#         stride = input_wh // grid_size # 413 // 13 ...
#         anchors_per_out = det.shape[1]//(5+num_classes) # = 3
        
#         det = det.view(batch, anchors_per_out*(5+num_classes), -1)
#         det = det.transpose(1, 2).contiguous()
#         det = det.view(batch, -1, (5+num_classes))
#         anchors = [ [a[0]/stride, a[1]/stride] for a in all_anchors[grid_size]] # convert amchor sizes into local scale
#         anchors = torch.cuda.FloatTensor(anchors)

#         # det: [batch, grid*grid*anchor, 5+classes] = [cx, cy, w, h, objectness, (class)]
#         det[:,:,0] = torch.sigmoid(det[:,:,0]) # centerX
#         det[:,:,1] = torch.sigmoid(det[:,:,1]) # centerY
#         det[:,:,2] = torch.exp(det[:,:,2]) # w
#         det[:,:,3] = torch.exp(det[:,:,3]) # h
#         det[:,:,4] = torch.sigmoid(det[:,:,4]) # objectness
#         det[:,:,5:] = torch.sigmoid(det[:,:,5:]) # class probability

#         x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
#         x, y = torch.cuda.FloatTensor(x).view(-1,1), torch.cuda.FloatTensor(y).view(-1,1)
#         xy_grid = torch.cat((x,y),1).repeat(1,anchors_per_out).view(-1,2).unsqueeze(0) # [1, 13*13*3, 2]
#         det[:,:,:2] += xy_grid # :: b = sigmoid(prediction) + x

#         anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0) # [1, 13*13*3, 2]
#         det[:,:, 2:4] *= anchors  # :: exp(w)*p, exp(h)*p
#         det[:,:,:4] *= stride  # get coordinates to original image scale
#         new_detections.append(det)
#     return new_detections


def convert_scores_to_sigmoid(detections, num_classes):
    """
    detections: list [[b, (5+classes), h, w], [b, (5+classes), h, w], [b, (5+classes), h, w]]
    num_classes: number of classes to detect
    """
    new_detections = list()
    for i in range(len(detections)):
        det = detections[i]
        batch = det.shape[0]
        num_local_anchors = det.shape[1]//(5+num_classes) # = 3
        
        det = det.view(batch, num_local_anchors*(5+num_classes), -1)
        det = det.transpose(1, 2).contiguous()
        det = det.view(batch, -1, (5+num_classes))

        # det: [batch, grid*grid*anchor, 5+classes] = [cx, cy, w, h, objectness, (class)]
        det[:,:,4] = torch.sigmoid(det[:,:,4]) # objectness
        det[:,:,5:] = torch.sigmoid(det[:,:,5:]) # class probability
        new_detections.append(det)
    return new_detections


def convert_coordinates_to_offset(detections, input_size, num_classes, grids, all_anchors):
    """
    detections: list, [[b, all_anchors, (5+classes)], ...]
    """
    new_detections = list()
    for i in range(len(detections)):
        det = detections[i]
        
        batch = det.shape[0]
        grid_size = grids[i][0] # w or h
        stride = input_size // grid_size # 413 // 13 ...
        num_local_anchors = det.shape[1]
        local_anchors = [ [a[0]/stride, a[1]/stride] for a in all_anchors[grid_size]] # convert original anchor sizes into local scale
        local_anchors = torch.cuda.FloatTensor(local_anchors)

        print("det", det.shape, grid_size, stride, num_local_anchors)

        # det: [batch, grid*grid*anchor, 5+classes] = [cx, cy, w, h, objectness, (class)]
        det[:,:,0] = torch.sigmoid(det[:,:,0]) # centerX
        det[:,:,1] = torch.sigmoid(det[:,:,1]) # centerY
        det[:,:,2] = torch.exp(det[:,:,2]) # w
        det[:,:,3] = torch.exp(det[:,:,3]) # h

        x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        x, y = torch.cuda.FloatTensor(x).view(-1,1), torch.cuda.FloatTensor(y).view(-1,1)
        xy_grid = torch.cat((x,y),1).repeat(1, num_local_anchors//(grid_size**2)).view(-1,2).unsqueeze(0) # [1, 13*13*3, 2]
        det[:,:,:2] += xy_grid # :: b = sigmoid(prediction_x) + x

        local_anchors = local_anchors.repeat(grid_size*grid_size, 1).unsqueeze(0) # [1, 13*13*3, 2]
        det[:,:, 2:4] *= local_anchors  # :: exp(prediction_w)*p, exp(prediction_h)*p
        det[:,:,:4] *= stride  # convert lovcal scale coordinates to original image scale
        new_detections.append(det)
    return new_detections




def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    # print(box_a)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def trans_anchors(anchors):
    new_anchors = torch.zeros((anchors.size(0), 4))
    new_anchors[:, :2] += 2000
    new_anchors[:, 2:] = anchors[:,]
    return point_form(new_anchors)

def trans_truths(truths):
    new_truths = torch.zeros((truths.size(0), 4))
    new_truths[:, :2] += 2000
    new_truths[:, 2:] = truths[:, 2:4]
    return point_form(new_truths)


def encode_targets_all(input_wh, truths, labels, best_anchor_idx, anchors, feature_dim, num_pred, back_mask):
    scale = torch.ones(num_pred).cuda()
    encode_truths = torch.zeros((num_pred, 6)).cuda()
    fore_mask = torch.zeros(num_pred).cuda()
    # l_dim, m_dim, h_dim = feature_dim
    l_grid_wh, m_grid_wh, h_grid_wh = feature_dim
    for i in range(best_anchor_idx.size(0)):
        index = 0
        grid_wh = (0, 0)
        # mask [0, 1, 2]
        if best_anchor_idx[i].item() < 2.1:
            grid_wh = l_grid_wh
            index_begin = 0
        # mask [3, 4, 5]
        elif best_anchor_idx[i].item() < 5.1:
            grid_wh = m_grid_wh
            index_begin = l_grid_wh[0] * l_grid_wh[1] * 3
        # mask [6, 7, 8]
        else:
            grid_wh = h_grid_wh
            index_begin = (l_grid_wh[0]*l_grid_wh[1] + m_grid_wh[0]*m_grid_wh[1]) * 3
        x = (truths[i][0] / input_wh[0]) * grid_wh[0]  
        y = (truths[i][1] / input_wh[1]) * grid_wh[1]
        floor_x, floor_y = math.floor(x), math.floor(y)
        anchor_idx = best_anchor_idx[i].int().item() % 3
        index = index_begin + floor_y * grid_wh[0] * 3 + floor_x * 3 + anchor_idx

        scale[index] = scale[index] + 1. - (truths[i][2] / input_wh[0]) * (truths[i][3] / input_wh[1])

        # encode targets x, y, w, h, objectness, class
        truths[i][0] = x - floor_x
        truths[i][1] = y - floor_y
        truths[i][2] = torch.log(truths[i][2] / anchors[best_anchor_idx[i]][0] + 1e-8)
        truths[i][3] = torch.log(truths[i][3] / anchors[best_anchor_idx[i]][1] + 1e-8)
        encode_truths[index, :4] = truths[i]
        encode_truths[index, 4] = 1.
        encode_truths[index, 5] = labels[i].int().item()

        # set foreground mask to 1 and background mask to 0, because  pred should have unique target
        fore_mask[index] = 1.
        back_mask[index] = 0
    return encode_truths, fore_mask > 0, scale, back_mask


def targets_match_all(input_wh, pos_iou_threshold, targets, prediction, anchors, feature_dim, pred_t, scale_t, fore_mask_t, back_mask_t, num_pred, idx, cuda=True):
    """
    targets: [num_anchors, 5] (x1, y1, x2, y2)
    pred: [num_anchors, 4] (cx, cy, w, h)
    """
    prediction = prediction.data.cpu()
    loc_truths = targets[:, :4].data # [num_anchors, 4]
    labels = targets[:,-1].data # [num_anchors, 1]
    print("cuda", loc_truths.is_cuda, prediction.is_cuda)
    overlaps = jaccard(
        loc_truths, 
        point_form(prediction)) # []
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    back_mask = (best_truth_overlap - pos_iou_threshold) < 0

    anchors = torch.FloatTensor(anchors)    
    if cuda:
        anchors = anchors.cuda()

    center_truths = center_size(loc_truths)
    new_anchors = trans_anchors(anchors)
    new_truths = trans_truths(center_truths)
    overlaps_ = jaccard(
        new_truths,
        new_anchors)
    best_anchor_overlap, best_anchor_idx = overlaps_.max(1, keepdim=True)
    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)

    encode_truths, fore_mask, scale, back_mask = encode_targets_all(input_wh, center_truths, labels, best_anchor_idx, anchors, feature_dim, num_pred, back_mask)

    pred_t[idx] = encode_truths
    scale_t[idx] = scale
    fore_mask_t[idx] = fore_mask
    back_mask_t[idx] = back_mask


def decode(prediction, input_wh, anchors, num_classes, stride_wh, cuda=True):
    grid_wh = (input_wh[0] // stride_wh[0], input_wh[1] // stride_wh[1])
    grid_w = np.arange(grid_wh[0])
    grid_h = np.arange(grid_wh[1])
    a,b = np.meshgrid(grid_w, grid_h)    

    num_anchors = len(anchors)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    anchors = [(a[0]/stride_wh[0], a[1]/stride_wh[1]) for a in anchors]
    if cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    print("offset: ", grid_wh, grid_w, grid_h, prediction.shape, x_y_offset.shape)
    prediction[:,:,:2] += x_y_offset
    anchors = torch.FloatTensor(anchors)
    if cuda:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_wh[0]*grid_wh[1], 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors
    prediction[:,:,0] *= stride_wh[0]
    prediction[:,:,2] *= stride_wh[0]
    prediction[:,:,1] *= stride_wh[1]
    prediction[:,:,3] *= stride_wh[1]
    return prediction


def permute_sigmoid(x, input_wh, num_anchors, num_classes):
    batch_size = x.size(0)
    grid_wh = (x.size(3), x.size(2))
    input_w, input_h = input_wh
    stride_wh = (input_w // grid_wh[0], input_h // grid_wh[1])
    bbox_attrs = 5 + num_classes
    x = x.view(batch_size, bbox_attrs*num_anchors, grid_wh[0] * grid_wh[1])
    x = x.transpose(1,2).contiguous()
    x = x.view(batch_size, grid_wh[0]*grid_wh[1]*num_anchors, bbox_attrs)
    x[:,:,0] = torch.sigmoid(x[:,:,0])
    x[:,:,1] = torch.sigmoid(x[:,:,1])             
    x[:,:, 4 : bbox_attrs] = torch.sigmoid((x[:,:, 4 : bbox_attrs]))
    return x, stride_wh