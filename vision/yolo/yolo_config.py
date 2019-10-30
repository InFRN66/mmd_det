import numpy as np
import torch
import torch.nn as nn


class Config():
    def __init__(self):
        self.image_size = 416
        self.input_wh = (416, 416)
        self.blocks = [1,2,8,8,4]
        self.num_classes = 80
        self.pos_iou_threshold = 0.5
        self.grids = ((13,13), (26,26), (52,52))
        self.all_anchors = {
            13: [[116, 90], [156, 198], [373, 326]],
            26: [[30, 61], [62, 45], [59, 119]],
            52: [[10, 13], [16, 30], [33, 23]]
            }  # grid_size: [[anchors]...]

    
        