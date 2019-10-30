import torch
import numpy as np
import pickle

COCO_CLASSES = ('backgrounds', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


VOC_CLASSES = ('backgrounds', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')

def IoU(bb1, bb2):
    '''
    IoUを計算，return
    bb1 : [x1,y1,x2,y2]
    bb2 : [x1,y1,x2,y2]
    '''
    bb1_area = ((bb1[2]-bb1[0]).item()) * ((bb1[3]-bb1[1]).item())
    bb2_area = ((bb2[2]-bb2[0]).item()) * ((bb2[3]-bb2[1]).item())
    cross_x1 = max(bb1[0].item(), bb2[0].item())
    cross_y1 = max(bb1[1].item(), bb2[1].item())
    cross_x2 = min(bb1[2].item(), bb2[2].item())
    cross_y2 = min(bb1[3].item(), bb2[3].item())

    if cross_x2-cross_x1 < 0.0 or cross_y2-cross_y1 < 0.0:
        print('Not overlapped')
        return 0.0

    overlap_area = (cross_x2-cross_x1) * (cross_y2-cross_y1)
    iou_score = (overlap_area) / (bb1_area + bb2_area - overlap_area)
    return iou_score
