"""
Pick up [num_images] images randomly, and 
analyze how much anchors are selected as positive samples in the traingng process.
Save result in ./visualize_anchors_overlap_shapes/statics_anchors.csv

default: 
    dataset: VOC-07-train + VOC-12-train
    model: vgg16-ssd
    Terms(per one object, not per one image):
        [gt_sizes]: area pf gt box, in normalized form (0-1)
        [gt_coord]: coordinates of gt box, in normalized form (0-1)
        [num_anchors]: num anchors which selected as positive samples in the object
        [indicies]: index of each [num_anchors] in priors (e.g. 0-8732)
Usage:
    python run_statics.py --seed 123 --file ./statics_anchors123.csv
"""

import numpy as np
import torch
import cv2
import os
import sys
import random
sys.path.append("../")
import argparse

from vision.ssd.ssd import MatchPrior
from vision.utils import box_utils
from vision.utils.box_utils import (assign_priors, center_form_to_corner_form, 
                                    corner_form_to_center_form, iou_of, for_statics_anchors)
from vision.ssd.config.vgg_ssd_config import Config

from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.ssd import MatchPrior
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from torch.utils.data import DataLoader, Dataset, ConcatDataset

parser = argparse.ArgumentParser(description="options")
parser.add_argument("--seed") # random seed.
parser.add_argument("--iou_thresh", type=float) # threshold for iou.
parser.add_argument("--path") # path for save csv files.
parser.add_argument("--num_imgs", type=int, default=1000) # num vimages to analyse.
args = parser.parse_args()

# --- seed setup
SEED = int(args.seed)

# --- torch random
torch.manual_seed(SEED)
# --- torch cudnn random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# --- default random
random.seed(SEED)
# --- numpy random
np.random.seed(SEED)
# --- os random 
os.environ["PYTHONHASHSEED"] = str(SEED)

anchors = [1,1,1]
rectangles = [[2],[2,3],[2,3],[2,3],[2],[2]]
num_anchors = [4,6,6,6,4,4]
vgg_config=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
config = Config(anchors, rectangles, num_anchors, vgg_config)

train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance,
                              iou_threshold=args.iou_thresh, path=args.path, stat=True)

def main():
    dataset01 = VOCDataset("/mnt/hdd01/img_data/VOCdevkit/VOC2007", 
                            transform=train_transform, target_transform=target_transform)
    dataset02 = VOCDataset("/mnt/hdd01/img_data/VOCdevkit/VOC2012", 
                            transform=train_transform, target_transform=target_transform)
    train_dataset = ConcatDataset([dataset01, dataset02])
    stat_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=1)

    for i, _ in enumerate(stat_loader):
        if i % 100 == 0:
            print(f"@{i+1} / {args.num_imgs}")
        if i+1 == args.num_imgs:
            break
    
if __name__ == "__main__":
    main()