import numpy as np
import torch
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import sys, os, argparse, glob, pickle, json
import numpy as np

from vision.datasets.voc_dataset import VOCDataset

'''
my own transform for scaling flicker experiments.
'''
    
def box_translation(boxes, h, w):
    """
    boxを0-1のスケールに変換
    boxes : [b, num_bb, 4]
    """
    assert (np.abs(boxes)).max() > 1.0
    boxes = boxes.astype(np.float32)
    norm_boxes = np.zeros_like(boxes)
    for i in range(len(boxes)):
        norm_boxes[i, :, 0] = boxes[i, :, 0] / w
        norm_boxes[i, :, 2] = boxes[i, :, 2] / w # x
        norm_boxes[i, :, 1] = boxes[i, :, 1] / h 
        norm_boxes[i, :, 3] = boxes[i, :, 3] / h # y
    return norm_boxes
    
    
# class org2square:
#     def __init__(self):
#         self.size = 300
        
#     def __call__(self, img):
#         """
#         Args :
#             img : [h, w, c]
#         """
#         return cv2.resize(img, (self.size, self.size))
    

def scale_transform(img, boxes, scale=0.95, x_shift=0, y_shift=0):
    """
    Args : 
        img : image / [h, w, c] , 
        boxes : bb represented in the img scale. [num_bb, 4]  
    """
    if type(img) == torch.Tensor:
        img = img.numpy()
    if img.ndim != 3:
        img = img[0]
    if type(boxes) == torch.Tensor:
        boxes = boxes.numpy()
    if boxes.ndim != 2:
        boxes = boxes[0]
    
    new_boxes = np.zeros_like(boxes)
    bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]) for bb in boxes], 
                                  shape=img.shape)
    
    if scale >= 1.0: # shift + 拡大
        seq = iaa.Sequential([
            iaa.Affine(translate_px={'x':x_shift, 'y':y_shift}, mode='edge'),
            iaa.Affine(scale=(scale, scale), mode='edge'),
            ])
    elif scale < 1.0: # 縮小 + shift
        seq = iaa.Sequential([
            iaa.Affine(scale=(scale, scale), mode='edge'),
            iaa.Affine(translate_px={'x':x_shift, 'y':y_shift}, mode='edge'),
            ])

    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        new_boxes[i, 0] = after.x1_int
        new_boxes[i, 1] = after.y1_int
        new_boxes[i, 2] = after.x2_int
        new_boxes[i, 3] = after.y2_int
        
    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(img, thickness=3, color=[255, 0,0])
    image_after = bbs_aug.draw_on_image(img_aug, thickness=3, color=[0, 0, 255])
    
#     return new_boxes, image_before, image_after, img_aug
    return img_aug, new_boxes

    
class ScaleTrans:
    """
    compress original image with `scale_coeff`, augument to `num_augment`
    """
    def __init__(self, num_augment, scale_coeff, transform=scale_transform):
        self.num_augment = num_augment
        self.transform = transform
        self.scale_coeff = scale_coeff
        
    def __call__(self, org_img, org_boxes, DIFFXs, DIFFYs):
        img_list = list()
        boxes_list = list()
        for i in range(self.num_augment):
            img, boxes = self.transform(org_img, org_boxes, scale=self.scale_coeff**i, x_shift=DIFFXs[i], y_shift=DIFFYs[i])
            img_list.append(img[np.newaxis, :, :, :])
            boxes_list.append(boxes[np.newaxis, :, :])
        img = np.vstack(img_list)
        boxes = np.vstack(boxes_list)
        
        return img, boxes



class GridTrans:
    '''
    並行移動にaugment変換
    '''
    def __init__(self, shift_range):
        '''
        Args:
            shift_range: (int) １画像ごとに何ピクセル動かすか
        '''
        self.shift_range = shift_range
        
    def __call__(self, images, boxes, labels, x_shift, y_shift):
        assert images.ndim == 3 and images.shape[-1] == 3
        assert images.min() >= 0 and images.max() <= 255
        # print(boxes.shape, labels)
        assert boxes.ndim == 2 and boxes.shape[0] == 1 # [n_box, 4], 1batch
        bb = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=boxes[0,0], y1=boxes[0,1], x2=boxes[0,2], y2=boxes[0,3]),
            ], shape=images.shape)
        # アフィン変換を定義
        # aug = iaa.Affine(translate_percent={'x':30, 'y':20})
        aug = iaa.Affine(translate_px={'x':x_shift, 'y':y_shift}, mode='edge')
        # 画像とボックスを変換
        aug_img = aug.augment_image(images)
        aug_bb = aug.augment_bounding_boxes([bb])[0]
        aug_bb = aug_bb.bounding_boxes[0]
        aug_bb = np.array([[aug_bb.x1_int, aug_bb.y1_int, aug_bb.x2_int, aug_bb.y2_int]])
        
        return aug_img, aug_bb, labels
    
    def augment(self, images, boxes, labels, direction, num=10):
        '''
        一枚の画像を複数シフト画像にaugment
        Args: 
            images: 入力画像，[h, w, c]
            boxes: 正解ボックス座標 [numbox(1), 4]
            label: 正解ラベル [numbox]
            direction: xyどちらにシフトするか．x[1,0], y[0,1], xy[1,1]
        Return:
            拡張画像群[b, h, w, c]
            対応ボックス座標[b, 1, 4]
            対応ラベル　[b] (ラベルは変化しないので同じ値)
        '''
        IMG, BOX, LABEL = [ list() for _ in range(3)]
        for i in range(num):
            img, b, lab = self(images, boxes, labels, x_shift=i*direction[0]*self.shift_range, y_shift=i*direction[1]*self.shift_range)
            IMG.append(img)
            BOX.append(b)
            LABEL.append(lab)
            
        return np.stack(IMG), np.stack(BOX), np.stack(LABEL)


class AnchorTrans:
    '''
    アンカー（アスペクト比）にaugment変換
    adjust_centerと併用でボックスの位置gridをそのまま，アンカーだけ変化
    '''
    def __init__(self, x_scale, y_scale):
        '''
        Args:
            x_ scale, y_scale:(float) １変換でxyをどれだけ圧縮（引き延ばし）するか
        '''
        self.x_scale = x_scale
        self.y_scale = y_scale
        
        
    def __call__(self, images, boxes, labels, x_scale, y_scale):
        assert images.ndim == 3 and images.shape[-1] == 3
        assert images.min() >= 0 and images.max() <= 255
        assert boxes.ndim == 2 and boxes.shape[0] == 1 # [n_box, 4], 1batch
        bb = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=boxes[0,0], y1=boxes[0,1], x2=boxes[0,2], y2=boxes[0,3]),
            ], shape=images.shape)
        # アフィン変換を定義(anchor)
        aug = iaa.Affine(scale={'x':x_scale, 'y':y_scale}, mode='edge')

        # aug = iaa.Sequential([
        #     iaa.Affine(scale=(scale, scale), mode='edge'),
        #     iaa.Affine(translate_px={'x':x_shift, 'y':y_shift}, mode='edge'),
        #     ])

        
        # 画像とボックスを変換
        aug_img = aug.augment_image(images)
        aug_bb = aug.augment_bounding_boxes([bb])[0]
        aug_bb = aug_bb.bounding_boxes[0]
        aug_bb = np.array([[aug_bb.x1_int, aug_bb.y1_int, aug_bb.x2_int, aug_bb.y2_int]])
        return aug_img, aug_bb, labels
    
    
    def augment(self, images, boxes, labels, num=10):
        '''
        一枚の画像を複数シフト画像にaugment
        Args: 
            images: 入力画像，[h, w, c]
            boxes: 正解ボックス座標 [numbox(1), 4]
            labels: 正解ラベル [numbox]
            num: 合計拡張枚数
        Return:
            拡張画像群[b, h, w, c]
            対応ボックス座標[b, 1, 4]
            対応ラベル　[b] (ラベルは変化しないので同じ値)
        '''
        IMG, BOX, LABEL = [ list() for _ in range(3)]
        for i in range(num):
            img, b, lab = self(images, boxes, labels, x_scale=(self.x_scale)**i, y_scale=(self.y_scale)**i)
            IMG.append(img)
            BOX.append(b)
            LABEL.append(lab)
            
        return np.stack(IMG), np.stack(BOX), np.stack(LABEL)


class ResizeTrans:
    '''
    画像のリサイズ
    '''
    def __init__(self, size):
        '''
        Args: 
            size: the size of iamge to transform ex) 300
        '''
        if type(size) == int:
            self.h_size = size
            self.w_size = size
        else:
            self.h_size = size[0]
            self.w_size = size[1] # size = [h, w]
        
    def __call__(self, imgs, boxes):
        """
        Args : boxes / normalized boxes [num, 4] (0 ~ 1)
            imgs / [num, h, w, c]
        """
        out_imgs = list()

        new_boxes = np.zeros_like(boxes)
        for i in range(len(boxes)):
            out_imgs.append(cv2.resize(imgs[i], (self.w_size, self.h_size))[np.newaxis, :, :, :])
            new_boxes[i, :, 0] = boxes[i, :, 0] * self.w_size
            new_boxes[i, :, 2] = boxes[i, :, 2] * self.w_size
            new_boxes[i, :, 1] = boxes[i, :, 1] * self.h_size
            new_boxes[i, :, 3] = boxes[i, :, 3] * self.h_size
        return np.vstack(out_imgs), new_boxes
        

class SubMeans:
    '''
    画像の平均値を引く
    '''
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, imgs):
        for i in range(len(imgs)):
            imgs[i, :, :, :] -= self.mean.astype(np.float32)
        return imgs

    
class My_Dataset(VOCDataset):
    # VOCDatasetを引き継ぎ，getitemだけtransformを変更
    def __getitem__(self, index):
        """
        alg:
            img = org [h, w] / h,w
            new_img = scaling(img) /  

            random sample random mirror
            resize 
            sub mean 
        """ 
        image_id = self.ids[index]
        org_boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            org_boxes = org_boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        org_img = self._read_image(image_id) # [h, w, c]
        
        return org_img, org_boxes, labels


def generate_heatmap(img, grid_size, place, RGB=True):
    '''
    Args:
        img: [h,w,c], 重ねたい画像 (300,300) 
        grid_size: list or tuple, featuremapの画素数
        place: 255にする画素位置, [h, w]
    return:
        [h, w, c] heatmap + image (300,300) 
    '''
    org_size = img.shape[:2]
    cmap = np.zeros((grid_size[0], grid_size[1]))
    cmap[place[0], place[1]] = 255
    cmap = cv2.applyColorMap(cmap.astype(np.uint8), cv2.COLORMAP_HOT)
    cmap = cv2.resize(cmap, (org_size[1], org_size[0]))
    
    print(img.shape, cmap.shape)
    assert img.shape == cmap.shape
    if RGB:
        return cv2.addWeighted(img, 0.8, cmap, 0.8, 0.5)[:,:,::-1]
    
    return cv2.addWeighted(img, 0.8, cmap, 0.5, 0.5)
        
        
def adjust_center(images, boxes, labels, dx=None, dy=None):
    '''
    ボックスが中央に来るように変換
    '''
    assert images.ndim == 3 and images.shape[-1] == 3
    assert images.min() >= 0 and images.max() <= 255
    assert boxes.ndim == 2 and boxes.shape[0] == 1 # [n_box, 4], 1batch

    if dx == None or dy == None:    
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis, :] # [1,4]
        chb = (boxes[0,3] + boxes[0,1])/2
        cwb = (boxes[0,2] + boxes[0,0])/2
        h, w = images.shape[:2]
        dy = int(h/2 - chb)
        dx = int(w/2 - cwb) # 中央に至るまでどれだけボックスを動かしたか    
    
    bb = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=boxes[0,0], y1=boxes[0,1], x2=boxes[0,2], y2=boxes[0,3]),
        ], shape=images.shape)
    # アフィン変換を定義(anchor)
    print('inner _dxdy', dx, dy)
    aug = iaa.Affine(translate_px={'x':dx, 'y':dy})
    # 画像とボックスを変換
    aug_img = aug.augment_image(images)
    aug_bb = aug.augment_bounding_boxes([bb])[0]
    aug_bb = aug_bb.bounding_boxes[0]
    aug_bb = np.array([[aug_bb.x1_int, aug_bb.y1_int, aug_bb.x2_int, aug_bb.y2_int]])
    return aug_img, aug_bb, labels, dx, dy

    
#     def calc_center_diff(self, images, boxes, labels):
#         assert images.ndim == 3 and images.shape[-1] == 3
#         assert images.min() >= 0 and images.max() <= 255
#         assert boxes.ndim == 2 and boxes.shape[0] == 1 # [n_box, 4], 1batch
        
#         if boxes.ndim == 1:
#             boxse = boxes[np.newaxis, :] # [1,4]
#         chb = (boxes[0,3] + boxes[0,1])/2
#         cwb = (boxes[0,2] + boxes[0,0])/2
#         h, w = images.shape[:2]
#         diff_h = int(h/2 - chb)
#         diff_w = int(w/2 - cwb)
# #         print(diff_h, diff_w)
        
#         return (diff_w, diff_h)


# size: 出力画像サイズ, center: 元画像における，中心に据える画像ピクセル位置
def trans_shift(var_imgs, var_boxes, org_cx, org_cy, interploate_method=cv2.INTER_LINEAR):
    """
    var_imgs: [batch, h, w, c]
    var_boxes: [batch, 1, 4]
    org_cx, org_cy: box center of original box    
    """
    IMAGES, BOXES = list(), list()
    
    for i in range(len(var_imgs)):
        img = var_imgs[i].copy()
        h, w, _ = img.shape
        box = var_boxes[i].copy()
        cx,cy = (box[0,2]+box[0,0])/2, (box[0,3]+box[0,1])/2
        diffx, diffy = org_cx-cx, org_cy-cy
        
        new_img = cv2.getRectSubPix(img, (img.shape[1],img.shape[0]), (int(w/2)-diffx, int(h/2)-diffy), \
                                    interploate_method)
        box[0] = box[0] + np.array([org_cx-cx, org_cy-cy, org_cx-cx, org_cy-cy])
        IMAGES.append(new_img[np.newaxis, :,:,:])
        BOXES.append(box[np.newaxis, :, :])
        
    return np.vstack(IMAGES), np.vstack(BOXES)
        