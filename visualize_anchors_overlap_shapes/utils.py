import numpy as np
import torch
import os, sys
import cv2
import matplotlib.pyplot as plt
sys.path.append("../")
from vision.ssd.ssd import MatchPrior
from vision.utils import box_utils
from vision.utils.box_utils import (assign_priors, center_form_to_corner_form, assign_priors_for_sigmoid_masking,
                                    corner_form_to_center_form, iou_of, for_statics_anchors)
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config.vgg_ssd_config import Config


anchors = [1,1,1]
rectangles = [[2],[2,3],[2,3],[2,3],[2],[2]]
num_anchors = [4,6,6,6,4,4]
vgg_config=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

config = Config(anchors, rectangles, num_anchors, vgg_config)
img_size = 300

match_prioir = MatchPrior(config.priors, config.center_variance, config.size_variance, iou_threshold=0.5)
corner_priors = center_form_to_corner_form(config.priors.clone())
center_priors = config.priors.clone()

alpha = 0.4
beta = 0.6
alpha_y = 0.001

fm = [38, 19, 10, 5, 3, 1]
anchor = [4, 6, 6, 6, 4, 4]
cuts = [0, 5776, 7942, 8542, 8692, 8728]


def bb_from_fourpoints(img_size, prior, color=(0,255,0), image=None):
    """
    view anrhor place in zero_image, from only index of prior
    """
    plt.figure(figsize=(10,10))
    if type(prior) == np.ndarray:
        prior = torch.from_numpy(prior)
    prior = prior.view(-1)
    if image is None:
        image = np.zeros((img_size, img_size, 3))
    image = cv2.rectangle(image, (prior[0], prior[1]),(prior[2], prior[3]), color, 1)
    plt.imshow(image)
    plt.show()
    return image


def point_anchor_center(priors_coordinates_cornerform, radian, color=(255,0,0), image=None):
    """
    point center point in featuremap
    """
    if image is None:
        image = np.zeros((img_size, img_size, 3))
    if priors_coordinates_cornerform.dim() != 2:
        priors_coordinates_cornerform.unsqueeze_(0)
    cx = (priors_coordinates_cornerform[0,2] + priors_coordinates_cornerform[0,0]) / 2
    cy = (priors_coordinates_cornerform[0,3] + priors_coordinates_cornerform[0,1]) / 2
    image = cv2.circle(image, (int(cx), int(cy)), radian, color, -1)
    return image


def bb_intersection_over_union(boxA, boxB):
    """
    get iou of two boxes
    boxA : [4]
    boxB : [4]
    """
    assert (boxA.mean() > 2) and (boxB.mean() > 2)
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(interArea) / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def for_iou(prior):
    """
    [8732, 4] -> [8732, 1, 4]
    """
    if prior.dim() != 3:
        prior = prior.unsqueeze(0).permute(1, 0, 2)
    return prior  


def overlap(img_size, box, thresh, sigmoid=None, image=None):
    if image is None:
        image = np.zeros((img_size, img_size, 3))
    else:
        img_size = image.shape[0]
    gt_labels = torch.Tensor([10])
    gt_boxes = torch.Tensor([
                             box
                            ]) # create input boxes to see
    gt_boxes_norm = gt_boxes / img_size
    for i in range(len(gt_boxes)):
        image = cv2.rectangle(image,(gt_boxes[i,0], gt_boxes[i,1]), (gt_boxes[i,2], gt_boxes[i,3]), (255,0,0), 2) 
    plt.figure(figsize=(10,10))
    plt.imshow(image/255)
    
    # --- check selected anchors ---
    if sigmoid:
        boxes, labels, mask = assign_priors_for_sigmoid_masking(gt_boxes_norm, gt_labels, corner_priors, alpha, beta, alpha_y)
    else:
        boxes, labels = assign_priors(gt_boxes_norm, gt_labels, corner_priors, iou_threshold=thresh)
        mask = None
    
    index_place = np.where(labels == 10)
    for prior in corner_priors[index_place] * img_size:
        image = cv2.rectangle(image, (prior[0], prior[1]),(prior[2], prior[3]),(0,255,255),1)
    print("positives: ", len(index_place[0]))
    print("index: ", index_place[0])
    plt.imshow(image/255)
    ious = iou_of(for_iou(corner_priors[index_place]), for_iou(gt_boxes_norm))
    return image, ious, mask


def get_one_scale_priors(priors, fm):
    """
    return one scale prior
    input: [8732, 4]
    output: list of each scale prior
    """
    if fm == 38:
        out = priors[:5776, :]
    elif fm == 19:
        out = priors[5776:7942, :]
    elif fm == 10:
        out = priors[7942:8542, :]
    elif fm == 5:
        out = priors[8542:8692, :]
    elif fm == 3:
        out = priors[8692:8728, :]
    elif fm == 1:
        out = priors[8728:, :]
    return out


def devide_priors_into_each_scales(priors):
    """
    return list of priors, in each scales
    """
    out = list()
    for i in fm:
        out.append(get_one_scale_priors(priors, i))
    return out


def res_to_ind(res):
    """
    return scale index, from resolution 
    """
    print(res)
    return {38:0, 19:1, 10:2, 5:3, 3:4, 1:5}[res]


def get_resolution(idx):
    """
    return resolution, from prior index
    """
    for i in range(len(cuts)-1):
        if (idx >= cuts[i]) and (idx < cuts[i+1]):
            return fm[i]
    return fm[i+1]

def torch_to_img(image):
    return image[0].permute(1,2,0).numpy()
        