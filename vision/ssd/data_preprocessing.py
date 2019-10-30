from ..transforms.transforms import *
from ..transforms.my_transform import *
from vision.utils.scale_change_utils import *

import numpy as np
import torch

class TrainAugmentation:

    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),      # from int to float
            PhotometricDistort(),   # 
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),      # from hw scale to norm scale
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image




# for scale various transform in retrain:
class Train_ScaleAugmentation_for_Retrain:
    def __init__(self, net_type, size, mean, std, cause_path, targets, mode='biggest'):
        """
        Args - net_type, cause_path
            net_type : type of net to use [vgg16-ssd, mb1-ssd, ...]
            cause_path : path to pickle file for analyze / ex) vgg16-ssd_cause.pickle / devided into scale, anchor, grid, non
            targets : scale, anchor, grid, non which of them will be extracted
            mode : (how to select target box) 'random'|random select / 'big'|select biggest box 
        """
        self.org_anchors, self.norm_anchors = return_fragments_boxes(cause_path, targets) # [num_fragments, 4], hw_scale, normalized_scale
        self.size = size
        self.mode = mode
        self.sub_mean = SubtractMeans(mean)
        self.dev_std = lambda img, boxes=None, labels=None: (img / std, boxes, labels)
        self.to_tensor = ToTensor()

    def __call__(self, img, boxes, labels):
        """
        Args img, boxes, labels
            img : [h, w, c], RGB
            boxes : [num_targets, 4] - hw_scale
            labels : [num_targets]
            *** std is not considerd here 'cause std == 1.0 ***
        """
        image, box, label, IOU, ret = self.search_matching_fragment_box(img, boxes, labels)
        image, box, label = self.sub_mean(image, box, label)
        image, box, label = self.dev_std(image, box, label)
        image, box, label = self.to_tensor(image, box, label)
        return image, box, label, IOU, ret

    
    def search_matching_fragment_box(self, img, boxes, labels):
        
        assert img.ndim == 3
        
        boxes = boxes.astype(np.float32)
        h, w, _ = img.shape
        norms = boxes.copy()
        norms[:, [0,2]] /= w
        norms[:, [1,3]] /= h

        num_use_targets = 1 # 1targetに対しscaleを探す
        if self.mode == 'random':
            select_target = np.random.randint(0, len(boxes), num_use_targets)[0]
        elif self.mode == 'biggest':
            areas = area_of(torch.from_numpy(boxes[:, :2]), torch.from_numpy(boxes[:, 2:]))
            select_target = areas.argmax()
        else:
            raise Exception(f'invalid mode {self.mode}')

        gt_box = boxes[[select_target], :] # ground truth [1, 4], hw scale
        norm_gt_box = norms[[select_target], :] # ground truth [1, 4], norm scale
        target_label = labels[[select_target]]

        IOU = iou_without_coordinates(self.norm_anchors, norm_gt_box) # compare 
        select_anchor = IOU.argmax()
        # print(f'best match fragment anchor IOU is {IOU[select_anchor]}')
        
        anchor = self.org_anchors[[select_anchor], :] # best match fragment error [1, 4] hw scale
        anchor_ = self.norm_anchors[[select_anchor], :] # best match fragment error [1, 4] norm scale

        # print('selected: ')
        # print('GT: ', gt_box)
        # print('AC: ', anchor) # normalize されてる
        # print()

        # # org_scaleで決める
        # # wの比で拡大率を決める
        # scale_coeff_w = (anchor[0, 2] - anchor[0, 0]) / (gt_box[0, 2] - gt_box[0, 0])
        # # hの比で拡大率を決める
        # scale_coeff_h = (anchor[0, 3] - anchor[0, 1]) / (gt_box[0, 3] - gt_box[0, 1])
        # scale_coeff = (scale_coeff_h + scale_coeff_w) / 2
        # # print('scale_coeff', scale_coeff)

        # norm_scaleで決める
        # wの比で拡大率を決める
        scale_coeff_w = (anchor_[0, 2] - anchor_[0, 0]) / (norm_gt_box[0, 2] - norm_gt_box[0, 0])
        # hの比で拡大率を決める
        scale_coeff_h = (anchor_[0, 3] - anchor_[0, 1]) / (norm_gt_box[0, 3] - norm_gt_box[0, 1])
        scale_coeff = min(scale_coeff_h, scale_coeff_w)


        post_trans = ScaleTrans(num_augment=2, scale_coeff=scale_coeff)
        var_imgs, var_boxes = post_trans(img, gt_box)
        # print('post0', var_imgs.shape)
        # print('post1', var_boxes.shape, var_boxes) # [num, 1, 4]

        # box normalize (scaleそた画像のboxをnormalize)
        norm_boxes = box_translation(var_boxes, h=h, w=w)
        # print('norm', norm_boxes.shape, norm_boxes) # [num, 1, 4]

        # resize (300)
        resize_trans = ResizeTrans(size=self.size)
        resized_imgs, resized_boxes = resize_trans(var_imgs, norm_boxes)
        # print('post3', resized_imgs.shape)
        # print('post4', resized_boxes.shape, boxes)

        # scaled_img = resized_imgs[1].transpose(2,0,1) # [3, 300, 300]
        scaled_img = resized_imgs[1] # [300, 300, 3]
        assert scaled_img.shape[-1] == 3

        # reshape from [num, 1, 4] to [num, 4], and clamp to 0 - 1
        if norm_boxes.ndim != 2:
            norm_boxes = norm_boxes.reshape(norm_boxes.shape[0], -1)
        norm_boxes = np.clip(norm_boxes, 0.0, 1.0)
        
        target_box = norm_boxes[[1], :]
        assert target_box.max() <= 1.0
        # return scaled_img, target_box, target_label

        if IOU.max() >= 0.9:
            ret = 1
        else:
            ret = 0
        # return resized_imgs, resized_boxes, (img, boxes, labels), scale_coeff, (anchor, anchor_), IOU.max()
        return scaled_img, target_box, target_label, IOU.max(), ret
