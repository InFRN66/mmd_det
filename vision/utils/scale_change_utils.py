import numpy as np
import os, sys, pickle, json
import numpy as np

from collections import defaultdict
import torch 

def boundary_rate(pickle_path):    
    with open(pickle_path, 'rb') as f:
        o = pickle.load(f)
    total = o['total']
    print('net_cause')
    print('scale : anchor : grid')
    print(f'{int(np.sum(total[:,0], axis=0))}/{len(total)} : {int(np.sum(total[:,1], axis=0))}/{len(total)} : {int(np.sum(total[:,2], axis=0))}/{len(total)} ')
    print(f'{int(np.sum(total[:,0], axis=0))/len(total):.3f} : {int(np.sum(total[:,1], axis=0))/len(total):.3f} : {int(np.sum(total[:,2], axis=0))/len(total):.3f} ')


def get_wh_ratio(boxes):
    """
    boxes : [N, 4]
    """
    wh = boxes[:, [2,3]] - boxes[:, [0,1]]
    wh_ratio = wh[:, 0] / wh[:, 1]
    return wh_ratio

def open_pickle(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


def get_fragment_frames(pickle_file, target=['scale', 'anchor', 'grid', 'non']):
    """
    return dict[file_keys][list of fragment frames]

    """
    interest_frames = defaultdict(list)
    for boundary in pickle_file.keys():
        # print(boundary)
        if boundary not in target:
            continue
        files = pickle_file[boundary]
        for key in files.keys():
            for i in range(len(files[key])):
                F = files[key][i][1]
                img_number = int(F.split('/')[-1].split('.')[0])
                # print(F, img_number)
                interest_frames[key].append(img_number)
            interest_frames[key].sort()
    return interest_frames


def return_gt_boxes(path, normalize=True):
    """
    return np.array[num_frames, 4]
    Args :: path : path to label.json
            normalize : if normalize boxes or not
    """
    with open(path, 'r') as f:
        gt = json.load(f)
    gt_boxes = []
    h, w = gt['h'], gt['w']
    # print(len(gt))
    for i in range(len(gt)-2): # remove h, w
        gt_box = np.array([[gt[str(i)]['x1'], gt[str(i)]['y1'], gt[str(i)]['x2'], gt[str(i)]['y2']]])
        if normalize:
            # print(gt_box)
            # print(np.array([[w,h,w,h]]))
            gt_box = gt_box / np.array([[w,h,w,h]])
        gt_boxes.append(gt_box)
    gt_boxes = np.vstack(gt_boxes)
    return gt_boxes
    
def area_of(left_top, right_bottom) -> torch.Tensor:
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]



def iou_without_coordinates(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args :
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    if boxes0.ndim != 2:
        boxes0 = boxes0[np.newaxis, :]
    if boxes1.ndim != 2:
        boxes1 = boxes1[np.newaxis, :]
    boxes0 = boxes0.astype(np.float64)
    boxes1 = boxes1.astype(np.float64)

    boxes0_woCorner = boxes0.copy()
    boxes0_woCorner[:, [0, 2]] -= np.repeat(boxes0_woCorner[:, [0]], 2, axis=1) # -x1
    boxes0_woCorner[:, [1, 3]] -= np.repeat(boxes0_woCorner[:, [1]], 2, axis=1) # -x1 
    
    boxes1_woCorner = boxes1.copy()
    boxes1_woCorner[:, [0, 2]] -= np.repeat(boxes1_woCorner[:, [0]], 2, axis=1) # -x1
    boxes1_woCorner[:, [1, 3]] -= np.repeat(boxes1_woCorner[:, [1]], 2, axis=1) # -x1 
    
    # print(boxes0_woCorner)
    # print(boxes1_woCorner)
    
    boxes0_woCorner = torch.from_numpy(boxes0_woCorner)
    boxes1_woCorner = torch.from_numpy(boxes1_woCorner)

    # print(boxes0_woCorner.type(), boxes1_woCorner.type())
    
    overlap_left_top = torch.max(boxes0_woCorner[..., :2].type(torch.DoubleTensor), boxes1_woCorner[..., :2].type(torch.DoubleTensor))
    overlap_right_bottom = torch.min(boxes0_woCorner[..., 2:].type(torch.DoubleTensor), boxes1_woCorner[..., 2:].type(torch.DoubleTensor))

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0_woCorner[..., :2], boxes0_woCorner[..., 2:])
    area1 = area_of(boxes1_woCorner[..., :2], boxes1_woCorner[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def return_fragments_boxes(cause_path, targets=['scale', 'anchor', 'grid', 'non']):
    vgg_pickle = open_pickle(cause_path)
    interest_frames = get_fragment_frames(vgg_pickle, target=targets)

    norm_boxes = []
    org_boxes = []
    for key in interest_frames.keys():
        frames = interest_frames[key]
        with open(os.path.join('/mnt/disk1/img_data/DAVIS/2017/DAVIS/Annotations/again', key, 'label.json'), 'r') as f:
            label = json.load(f)
        h, w = label['h'], label['w']
        frame_info = [ label[str(frame)] for frame in frames ]    
        assert len(frame_info) > 0
        
        # print(len(frame_info), frame_info[0])
        # print('img_hw: ', h, w)
        norm_box = [ np.array([frame_info[i]['x1']/w, frame_info[i]['y1']/h, frame_info[i]['x2']/w, frame_info[i]['y2']/h]) \
                    for i in range(len(frame_info))]
        norm_box = np.vstack(norm_box)
        assert norm_box.shape[-1] == 4 # norm_boxes
        norm_boxes.append(norm_box)

        org_box = [ np.array([frame_info[i]['x1'], frame_info[i]['y1'], frame_info[i]['x2'], frame_info[i]['y2']]) \
                    for i in range(len(frame_info))]
        org_box = np.vstack(org_box)
        assert org_box.shape[-1] == 4 # org_boxes
        org_boxes.append(org_box)
        
    norm_boxes = np.vstack(norm_boxes) # [num, 4]
    org_boxes = np.vstack(org_boxes)
    # box_w = (boxes[:, 2] - boxes[:, 0]) # w for each boxes
    # box_h = (boxes[:, 3] - boxes[:, 1]) # h for each boxes
    # box_hw = box_h / box_w # h-w ratio for each boxes
    return org_boxes, norm_boxes


# used for adding w and h to label.json
# import glob

# files = glob.glob('/mnt/disk1/img_data/DAVIS/2017/DAVIS/Annotations/self-train/*/')
# files.sort()
# for F in files:
#     h, w, _ = cv2.imread(os.path.join(F, '00000.png')).shape
#     with open(os.path.join(F, 'label.json'), 'r') as f0:
#         label = json.load(f0)
    
#     label['h'] = h
#     label['w'] = w
    
#     print(label.keys(), label['w'])
#     print()
    
#     with open(os.path.join(F, 'label.json'), 'w') as f1:
#         json.dump(label, f1)