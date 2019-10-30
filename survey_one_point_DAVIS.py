from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.resnet50_ssd import create_resnet50_ssd, create_resnet50_ssd_predictor
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

# config
from vision.ssd.config.vgg_ssd_config import Config
# from vision.ssd.config import resnet50_ssd_config
# from vision.ssd.config import mobilenetv1_ssd_config

import torch
from torch.utils.data import DataLoader
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import sys, os, argparse, glob, pickle, json
from collections import defaultdict, namedtuple
import time

from flicker_oneObject import Frame

from vision.datasets.voc_dataset import VOCDataset, ORG_Dataset
from vision.transforms.my_transform import *
import result_utils as ru


# --- to list from argparse
def to_list01(argument):
    f = lambda x: x.split(",")
    return list(map(int, f(argument)))

def to_list02(argument):
    out = list()
    f = lambda x: x.split(",")
    for x in f(argument):
        try:
            out.append(int(x))
        except ValueError:
            out.append(str(x))
    return out

parser = argparse.ArgumentParser(description="options")
parser.add_argument("--data_type", type=str, default="VOC", help="datasets to used for preprocess")
parser.add_argument("--image_dir", type=str, default=None, help="dir to load images")
parser.add_argument("--save_dir", type=str, default=None, help="dir to load annotation")
parser.add_argument("--fragment_file", type=str, default=None, help="file for DAVIS_flicker.json")
parser.add_argument("--confidence_threshold", default=0.50, type=float, help='Detection confidence threshold')
parser.add_argument("--IoU_threshold", default=0.50, type=float, help='Minimum IoU threshold for checking the same object')
parser.add_argument("--net_type", default="vgg16-ssd", type=str, help='model to use')
parser.add_argument("--target_DAVIS_flie_pickle", default="", type=str, help='model to use')
parser.add_argument("--change_factor", default="scale", type=str, help='[scale, (anchor), grid]')
parser.add_argument("--interpolate_method", default='inter-linear', type=str, help='cv2 interploation method for image transform')
parser.add_argument("--adjust_center", action='store_true', help="adjuct image transform so that the box center is not changed")
parser.add_argument("--label_path", default='models/voc-model-labels.txt', type=str, 
                    help='path to VOC label')
parser.add_argument("--model_weight", type=str, default="models/vgg16-ssd-mp-0_7726.pth", 
                    help="model weight path to use")

parser.add_argument("--fragment_info_dir", type=str, help="path to DAVOS_flicker.json", required=True)

parser.add_argument('--anchors', type=to_list01, help='selection useing anchor [small sq, big sq, rects]',
                    required=True)        
parser.add_argument('--rectangles', action="append", nargs="+", type=int, 
                    help='rectangles to use', required=True) 
parser.add_argument('--num_anchors', type=to_list01, help='num anchors in each fm', required=True) 
parser.add_argument('--vgg_config', type=to_list02, help='vgg network configuration', required=True) 

args = parser.parse_args()

net_type = args.net_type
model_path = args.model_weight
label_path = args.label_path

# make annotation path from imag_dir
try:
    image_dir = args.image_dir
    if args.data_type == 'DAVIS':
        pix, obj = image_dir.split('/')[-3:-1]
        paths = image_dir.split('/')[1:-4]
        paths.extend(['Annotations', pix, obj])
        annot_dir = os.path.join(*paths)
        annot_dir = '/'+annot_dir+'/label.json'
except:
    pass


if args.interpolate_method == 'inter-linear':
    interploate_method = cv2.INTER_LINEAR
elif args.interpolate_method == 'inter-nearest':
    interploate_method = cv2.INTER_NEAREST
elif args.interpolate_method == 'inter-cubic':
    interploate_method = cv2.INTER_CUBIC
else:
    raise Exception('invalid interpolation')


# fragment_file
# fragment_files = '/mnt/disk1/img_data/DAVIS/2017/DAVIS/JPEGImages/reTest/bike-packing/prob_percent_Diff0.4/'+net_type+'_0.5/DAVIS_flicker.json'
# fragment_files = '/mnt/disk1/img_data/DAVIS/2017/DAVIS/JPEGImages/self-train/bike-packing/prob_percent_Diff0.4/'+net_type+'_0.5/DAVIS_flicker.json'
if args.fragment_file:
    fragment_files = args.fragment_file


class_names = [name.strip() for name in open(label_path).readlines()]
rectangle_color = pickle.load(open("pallete", "rb"))

# hyper params
num_classes = len(class_names)
top_k = 200

# 2007
voc_root = '/mnt/hdd01/img_data/VOCdevkit/VOC2007'

# experiments setting
num_images = 1    # related only voc 
num_augment = 30  # num of transforms in per image

same_image_object_limit = 100 # 同じ画像で複数物体アノテーションされている場合，１０個とかは時間がかかるので使う物体の上限を設ける
track_index_limit = 100 # 同じ画像・同じ物体で追跡するbox(in 8732)の上限．多いとplot図が煩雑;


if args.change_factor == 'scale':
    # 拡大．縮小率
    # coeffs = [0.995, 1.005]
    coeffs = [1.01, 0.99]
    # coeffs = [0.98, 1.02]
    # coeffs = [1.05, 0.95]
    # coeffs = [1.0]
elif args.change_factor == 'grid':
    # 平行にずらすピクセル数
    coeffs = [3, -3]
    # coeffs = [5, -5]
    # coeffs = [10, -10]
    # -1ならx方向,-2ならy方向にシフト
    direction = [1,0] # x
    # direction = [0,1] # y
    # direction = [1,1] # xy

elif args.change_factor == 'anchor':
    # coeffs = [[0.98, round(1.0/0.98, 3)], [round(1.0/0.98, 3), 0.98]] # x, yへの圧縮（伸縮）倍率
    # coeffs = [[0.995, round(1.0/0.995, 3)], [round(1.0/0.995, 3), 0.995]]
    # coeffs = [[0.99, round(1.0/0.99, 3)], [round(1.0/0.99, 3), 0.99]]
    coeffs = [1.01, 0.99]

else:
    raise Exception('invalid factor')

# for backward
GRAD = []
SCALE = []

# store area and probability
AREAS = defaultdict(lambda: defaultdict(list))
PROBS_DICT = defaultdict(lambda: defaultdict(list))
LABELS = list()
NUM_BOX = 0

# COLORS = {38:'r', 19:'b', 10:'m', 5:'g', 3:'c', 2:'r', 1:'k'} # 2はmobileNet用
rectangle_color = [(255,0,0), (0,0,255), (255,0,255), (0,128,0), (0,255,255), (228,155,15), (0,0,0), \
                    (128,0,0), (0,0,128), (128,0,128), (0,64,0), (0, 255, 65), (0,128,128), (128,128,0), ] * 3
# COLORS = [('r', (255,0,0)), ('b',(0,0,255)), ('m', (255,0,255)), ('g', (0,128,0)), ('c', (0,255,255)), \
#              ('y', (255,255,0)), ('k', (0,0,0))] * 3
# GRID = ['o', 'P', '*', 'D', '^', '_', 's', '4', 'H', 'X', '|', 'v'] * 3
GRID = {38:'o', 19:'P', 10:'*', 5:'D', 3:'^', 2:'s', 1:'X'}
ANCHOR = ['o', 'P', '*', 'D', '^', 'X', '_', 's', '4', 'H',  '|', 'v'] * 3



def get_image_list(o):
    indicies = np.array(o['flicker_indicies'])
    indicies = np.where(indicies==1)[0]
    return indicies


def load_model(net_type):
    config = Config(args.anchors, args.rectangles, args.num_anchors, args.vgg_config)

    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), config, is_test=True)
    elif net_type == 'resnet50-ssd':
        net = create_resnet50_ssd(len(class_names), config, is_test=True)
    # elif net_type == 'mb1-ssd':
    #     print('mb1-ssd load')
    #     net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    # elif net_type == 'mb1-ssd-lite':
    #     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    # elif net_type == 'mb2-ssd-lite':
    #     net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    # elif net_type == 'sq-ssd-lite':
    #     net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200, config=config)
    elif net_type == 'resnet50-ssd':
        predictor = create_resnet50_ssd_predictor(net, candidate_size=200, config=config)
    # elif net_type == 'mb1-ssd':
    #     predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    #     config = mobilenetv1_ssd_config
    # elif net_type == 'mb1-ssd-lite':
    #     predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    # elif net_type == 'mb2-ssd-lite':
    #     predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    # elif net_type == 'sq-ssd-lite':
    #     predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    # else:
    #     predictor = create_vgg_ssd_predictor(net, candidate_size=200, config=config)
    return predictor, config


def detection_rectangle_write(image, dets, class_id, N, found):
    """
    detection boxを描画．
    image : [h,w,c], numpy, RGB
    dets : [num_bb, 5] conf,x1,y1,x2,y2 in one class, one image
    class_id : 1 ~ 80 or 1 ~ 20, because 0 is background and to be skipped
    """
    if found == 0:
        return image

    if len(dets) == 0:
        print("no detection was found.")
        return image

    for i in range(len(dets)):
        score = dets[i, 0]
        x1 = dets[i, 1]
        y1 = dets[i, 2]
        x2 = dets[i, 3]
        y2 = dets[i, 4]
        color = rectangle_color[N]
        # color = COLORS[N][1]
        label = class_names[int(class_id)] + ':' + str(round(score.item(), 4))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        # cv2.putText(image, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1.7, [0,0,0], 3) # 大きさ，太さ
        # cv2.putText(image, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1.7, [255,255,255], 1) # 大きさ，太さ
        cv2.putText(image, label, (x1+10, y1+t_size[1]+14), cv2.FONT_HERSHEY_PLAIN, 1.7, [0,0,0], 3) # 大きさ，太さ
        cv2.putText(image, label, (x1+10, y1+t_size[1]+14), cv2.FONT_HERSHEY_PLAIN, 1.7, [255,255,255], 1) # 大きさ，太さ
    return image


def true_rectangle_write(image, true_box, class_id, height, width):
    """
    正解ボックスを赤で描画．
    image : [h,w,c], numpy, RGB
    label : [num_bb, 5] conf,x1,y1,x2,y2 in one class, one image
    class_id : 1 ~ 80 or 1 ~ 20, because 0 is background and to be skipped
    """
    for i in range(len(true_box)):
        x1 = int(true_box[i, 0] * width)
        y1 = int(true_box[i, 1] * height)
        x2 = int(true_box[i, 2] * width)
        y2 = int(true_box[i, 3] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), 2, lineType=cv2.LINE_8)
    return image


# hook function
def grad_hook(self, grad_in, grad_out):
    # print('G_in', grad_in[0].shape, grad_in[1].shape, grad_in[2].shape, len(grad_in))
    # print('G_out', grad_out[0].shape, len(grad_out))
    place = np.where(grad_out[0].data.cpu().numpy()==1)
    if np.array(place).sum() > 0:
        GRAD.append(np.array(place))
        SCALE.append(list(grad_out[0].shape[1:]))


def main():
    # load net
    predictor, config = load_model(net_type)
    # hook for backward

    num_cl = len(list(predictor.net.classification_headers.children())) # 6
    for i in range(num_cl):
        predictor.net.classification_headers[i].register_backward_hook(grad_hook)

    # dataset
    if args.data_type == 'VOC':
        # SEED = 3
        SEED = 4
        np.random.seed(SEED)
        dataset = VOCDataset(root=voc_root, transform=None, target_transform=None, 
                             is_test=True, keep_difficult=False, return_id=True)
        random = np.random.randint(0, len(dataset), num_images)
        # random = np.array([10])


    elif args.data_type == 'DAVIS':
        # set1 : 任意のimageに対して実行
        dataset = ORG_Dataset(root=image_dir,
                              annot_root=annot_dir,
                              data_type=args.data_type)
        with open(os.path.join(Path(args.fragment_info_dir)/Path("DAVIS_flicker.json")), "r") as f:
            JSON = json.load(f)
        targets = JSON[Path(args.image_dir).name]["image_list"]
        random = np.array([int(x.split(".")[0]) for x in targets])

        # # ---
        # random = np.array([45])
        # # ---

        random_add = np.array([ [x - 1, x + 1] for x in random if x != 0 ])
        random_add = [flatten for inner in random_add for flatten in inner]
        random = np.array(random.tolist() + random_add)
        random = np.unique(random)
        

    print('random:', random)


    for j, i in enumerate(random):
        BOX_INFO = namedtuple('box', ('fm', 'anchor', 'grid_y', 'grid_x'))

        # 一枚の元画像
        if args.data_type == 'VOC':
            org_img, org_boxes, org_labels, img_id = dataset[i] # [h, w, c], [num_bb, 4], [num_bb]/ ボックスはhwスケール

        elif args.data_type == 'DAVIS':
            org_img, org_boxes, org_labels = dataset[i]
            img_id = i
        else:
            raise Exception('invalid type')

        # objectが単体の画像に絞るのも可

        # # 複数物体の扱い：ひとつに絞る， or 個別に走査する
        # if len(org_box) > 1:
        #     org_box = org_box[[0]]
        #     label = label[[0]]

        # 複数物体：個別にやる
        print(len(org_boxes), org_boxes)
        
        for b in range(len(org_boxes)):

            if b >= same_image_object_limit:
                break

            TRACK_BOX = list()
            IMAGE_STORE = dict()
            VAR_IMAGES = dict()
            NORM_BOXES = dict()

            org_box, label = org_boxes[[b]], org_labels[[b]]
            print('boxshape', org_box.shape, label.shape)
        
            save_dir = os.path.join(args.save_dir, Path(args.image_dir).name, str(img_id).zfill(6))
            os.makedirs(save_dir, exist_ok=True)

            for coeff, xaxis in zip(coeffs, [1, -1]): # up scale and down scale
                print('coeff', coeff, xaxis)
                if args.change_factor == 'scale':
                    # scaling
                    scale_trans = ScaleTrans(num_augment=num_augment, scale_coeff=coeff)
                    org_cx, org_cy = (org_box[0,2]+org_box[0,0])/2, (org_box[0,3]+org_box[0,1])/2
                    var_imgs, var_boxes = scale_trans(org_img, org_box, np.zeros(num_augment), np.zeros(num_augment))

                    if args.adjust_center:
                        DIFFX, DIFFY = list(), list()
                        for a in range(len(var_imgs)):
                            img = var_imgs[a].copy()
                            h, w, _ = img.shape
                            box = var_boxes[a].copy()
                            cx,cy = (box[0,2]+box[0,0])/2, (box[0,3]+box[0,1])/2 # (x2+x1)/2
                            diffx, diffy = org_cx-cx, org_cy-cy
                            DIFFX.append(diffx)
                            DIFFY.append(diffy)
                        DIFFX = np.array(DIFFX)
                        DIFFY = np.array(DIFFY)
                        var_imgs, var_boxes = scale_trans(org_img, org_box, DIFFX, DIFFY)

                    # if args.adjust_center:    
                    #     var_imgs, var_boxes = trans_shift(var_imgs, var_boxes, org_cx, org_cy, \
                    #                                         interploate_method=interploate_method)

                    # box normalize
                    norm_boxes = box_translation(var_boxes, h=org_img.shape[0], w=org_img.shape[1])
                    # resize (300)
                    resize_trans = ResizeTrans(size=300)
                    imgs, boxes = resize_trans(var_imgs, norm_boxes)
                    # sub mean
                    sub_trans = SubMeans(config.image_mean) # RBG value
                    imgs = sub_trans(imgs.astype(np.float32))
                    # in mb1, std is 128 (vgg is 1.0)
                    imgs /= config.image_std
                    # GPU cuda
                    imgs = imgs.transpose(0,3,1,2) # [b, c, 300, 300] maybe (b = 50とか，)
                    imgs = torch.from_numpy(imgs)
                    # imgs = imgs.cuda()


                elif args.change_factor == 'grid':
                    # 平行移動
                    grid_trans = GridTrans(shift_range=coeff)
                    var_imgs, var_boxes, labels = grid_trans.augment(org_img, org_box, label, direction=direction, num=num_augment)
                    # box normalize
                    norm_boxes = box_translation(var_boxes, h=org_img.shape[0], w=org_img.shape[1])
                    # resize (300)
                    resize_trans = ResizeTrans(size=300)
                    imgs, boxes = resize_trans(var_imgs, norm_boxes)
                    # sub mean
                    sub_trans = SubMeans(config.image_mean) # RBG value
                    imgs = sub_trans(imgs.astype(np.float32))
                    # in mb1, std is 128 (vgg is 1.0)
                    imgs /= config.image_std
                    # GPU cuda
                    imgs = imgs.transpose(0,3,1,2) # [b, c, 300, 300] maybe (b = 50とか)
                    imgs = torch.from_numpy(imgs)
                    # imgs = imgs.cuda()

                
                elif args.change_factor == 'anchor':
                    # # ボックスを中央に配置
                    # org_img, org_box, label, dx, dy = adjust_center(org_img, org_box, label)
                    # print('dx, dy', dx, dy)
                    # アスペクト比変更
                    anchor_trans = AnchorTrans(x_scale=coeff, y_scale=1/coeff)
                    org_cx, org_cy = (org_box[0,2]+org_box[0,0])/2, (org_box[0,3]+org_box[0,1])/2
                    var_imgs, var_boxes, labels = anchor_trans.augment(org_img, org_box, label, num=num_augment)
                    
                    # if args.adjust_center:
                    #     var_imgs, var_boxes = trans_shift(var_imgs, var_boxes, org_cx, org_cy, interploate_method=interploate_method)

                    # box normalize
                    norm_boxes = box_translation(var_boxes, h=org_img.shape[0], w=org_img.shape[1])
                    # resize (300)
                    resize_trans = ResizeTrans(size=300)
                    imgs, boxes = resize_trans(var_imgs, norm_boxes)
                    # sub mean
                    sub_trans = SubMeans(config.image_mean) # RBG value
                    imgs = sub_trans(imgs.astype(np.float32))
                    # in mb1, std is 128 (vgg is 1.0)
                    imgs /= config.image_std
                    # GPU cuda
                    imgs = imgs.transpose(0,3,1,2) # [b, c, 300, 300] maybe (b = 50とか)
                    imgs = torch.from_numpy(imgs)

                VAR_IMAGES[xaxis] = var_imgs
                NORM_BOXES[xaxis] = norm_boxes                
                frame = Frame(num_scale=num_augment, num_classes=21, num_bb=len(label),  \
                              h=var_imgs[0].shape[0], w=var_imgs[0].shape[1], c=var_imgs[0].shape[2], top_k=200)
                
                for s in range(len(imgs)):
                    print('s: ', s)
                    FACTOR = s # 0, 1, 2, 3, ,.
                    s = 0
                    written_image = None
                    for idx, c in enumerate(label): # label
                        # 上で既にsubmean resizeされてるのでpredictionTransformが不要
                        """
                        # detctions : [batch, 21, 200, 5]
                        # selected_indicies : [batch, 21, 200]
                        # row_scores : [batch, 8732, 21]
                        """
                        detections, selected_indicies, height, width, row_scores, _ = \
                                     predictor.predict_various_DAVIS(imgs[[FACTOR]].cuda(), height=org_img.shape[0], width=org_img.shape[1], 
                                                                     top_k=top_k, prob_threshold=args.confidence_threshold)
                        row_dets = detections[s, c, :] # [top_k, 5] = [200, 5]
                        print(row_dets)
                        
                        conf_thresh = args.confidence_threshold
                        IoU_thresh = args.IoU_threshold
                        ret = 0
                        
                        while ret == 0:
                            conf_mask = row_dets[:,0].gt(conf_thresh) # [top_k, 5]
                            try:
                                print('conf_mask =', conf_mask[:10, :])
                            except:
                                pass
                            
                            conf_mask = conf_mask.view(-1,1).expand_as(row_dets) # [top_k, 5]
                            dets = torch.masked_select(row_dets, conf_mask).view(-1, 5) # 0 - 1 scale
                            if dets.shape == torch.Size([0]): # detect結果無しの場合
                                
                                ret, found, sidx = frame.no_detection(s, c, idx)
                                conf_thresh *= 0.6
                                IoU_thresh *= 0.6
                                if conf_thresh <= 0.1 and ret == 0:
                                    print('forcibly turminate')
                                    ret = 1
                                    frame.track_box[s, c, idx, 1:] = frame.track_box[s-1, c, idx, 1:] # 前スケールのボックス | TODO:probは0でいいのか
                                    break
                                continue
                            
                            # print(s, c, idx, dets, norm_boxes[s, [idx]], IoU_thresh)
                            frame.class_act(s, c, idx, dets) # dets [num_bb, 5]
                            ret, found, sidx = frame.compare_IoU_one(s, c, idx, dets, norm_boxes[s, [idx]], IoU_thresh) # boxesもoriginal scale / normboxes [1, 4]

                            conf_thresh *= 0.6
                            IoU_thresh *= 0.6

                            if conf_thresh <= 0.1 and ret == 0:
                                print('forcibly terminate')
                                ret = 1
                                frame.track_box[s, c, idx, 1:] = frame.track_box[s-1, c, idx, 1:] # 前スケールのボックス

                    # backward processs:
                    target_index = selected_indicies[s, c, sidx].type(torch.LongTensor) # target index [batch, class, ]
                    TRACK_BOX.append(int(target_index))
                
                IMAGE_STORE[xaxis] = imgs.cuda()

            # print(len(IMAGE_STORE))
            # for k in IMAGE_STORE.keys():
            #     print(IMAGE_STORE[k].shape, IMAGE_STORE[k].is_cuda)
            if len(TRACK_BOX) == 0:
                break

            TRACK_BOX = torch.Tensor(TRACK_BOX).numpy()
            TRACK_BOX = np.unique(TRACK_BOX)
            if len(TRACK_BOX) >= track_index_limit:
                TRACK_BOX = TRACK_BOX[:track_index_limit]

            PLACE = dict()
            for key in TRACK_BOX:
                if key == 0:
                    continue
                target = row_scores[s, int(key), 0] # どこでもいいのでバックグラウンドクラスをバックプロップしても良い
                predictor.net.zero_grad()
                target.backward(retain_graph=True)
                try:
                    box_indicies = GRAD.pop() # [box_scale 中のグリッド位置]
                except:
                    print("no grad, exception.")
                    break
                # print('グリッド位置:', box_indicies.shape, box_indicies) # 判明している．
                box_scale = SCALE.pop() # box scale の種類

                # [どのfm, anchor, grid_y, grid_x]
                place_info = np.array([box_scale[-1], box_indicies.T[0, -3]//num_classes, box_indicies.T[0,-2], box_indicies.T[0, -1]])
                PLACE[int(key)] = place_info

                # BOX_INFO = namedtuple('box', ('fm', 'anchor', 'grid_y', 'grid_x'))
                PLACE[key] = BOX_INFO(box_scale[-1], box_indicies.T[0, -3]//num_classes, box_indicies.T[0,-2], box_indicies.T[0, -1])

            # del(detections, selected_indicies, height, width, row_scores)
            print('-----', PLACE)

            # time.sleep(20)
            # sys.exit(0) # # # ----------------------------------------------------------------------------------------------
            
            FACTOR_PROB = namedtuple('data_point', ('xaxis', 'factor', 'prob'))
            Data = defaultdict(list)
            for xaxis, coeff in zip(IMAGE_STORE.keys(), coeffs):

                print('---', xaxis, coeff)
                imgs = IMAGE_STORE[xaxis] # 片側transform [batch, 3, 300, 300], cuda=False
                var_imgs = VAR_IMAGES[xaxis]
                norm_boxes = NORM_BOXES[xaxis]

                for FACTOR in range(len(imgs)): # scale or grid # ここを繰り返す
                             
                    for idx, c in enumerate(label): # label
                        with torch.no_grad(): # gradは不要（バックプロップしない）のでオフにする：メモリ節約
                            row_conf, row_boxes, row_scores, height, width =  \
                                predictor.predict_oneBox(imgs[[FACTOR]].cuda(), height=org_img.shape[0], width=org_img.shape[1], 
                                                            top_k=top_k, prob_threshold=args.confidence_threshold)
                        
                        for N, track_index in enumerate(PLACE.keys()):
                            track_index = int(track_index)
                            # print(xaxis, FACTOR, track_index)
                            Data[track_index].append(FACTOR_PROB(xaxis, FACTOR, row_conf[s, track_index, c]))
                            prob = row_conf[0, track_index, c].view(1, -1).cpu() # [1,1]
                            coord = row_boxes[0, track_index, :].view(1, -1).cpu() # [1,4]
                            new_dets = torch.cat((prob, coord), 1) # [prob, x1, y1, x2, y2]
                            
                            new_dets[:,1] *= width
                            new_dets[:,2] *= height
                            new_dets[:,3] *= width
                            new_dets[:,4] *= height # detsはoriginal scale

                            # true box
                            # written_image = true_rectangle_write(var_imgs[FACTOR].copy(), norm_boxes[FACTOR], c, height, width)

                            # written_image = detection_rectangle_write(written_image, new_dets, c, N, found=1)

                            written_image = var_imgs[FACTOR].copy()

                            os.makedirs(os.path.join(save_dir,  str(track_index)), exist_ok=True)
                            Image.fromarray(np.uint8(written_image)).save(os.path.join(save_dir,  str(track_index), \
                                            f'{xaxis*FACTOR+num_augment}.jpg'))
                                # f'{args.change_factor}-[{coeff}*{FACTOR}n]-[{PLACE[track_index].fm}-{PLACE[track_index].anchor}-{PLACE[track_index].grid_y}-{PLACE[track_index].grid_x}].jpg'))
                                
                    # per label end #
                # per scale end


            # plot scatter | | 
            plt.rcParams['font.size'] = 12
            plt.rcParams['legend.fontsize'] = 10
            plt.rcParams['legend.handlelength'] = 2

            if args.change_factor == 'scale':    
                # Data : ('data_point', ('xaxis', 'factor', 'prob'))
                plt.figure(figsize=(12,6))
                for N, index in enumerate(Data.keys()):
                    data = np.array(Data[index])
                    plus = data[data[:,0] == 1]
                    plt.plot(plus[:, 1]*plus[:, 0]+num_augment, plus[:, 2], marker=GRID[PLACE[index].fm], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                            markersize=13, \
                                # label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]'
                                )
                    minas = data[data[:,0] == -1]
                    plt.plot(minas[:, 1]*minas[:, 0]+num_augment, minas[:, 2], marker=GRID[PLACE[index].fm], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                            markersize=13, \
                            label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]')
                plt.grid()
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
                plt.subplots_adjust(left=0.1, right=0.8)
                # plt.title(f'VOC_{img_id}_{args.change_factor}_{coeffs}')
                plt.xlabel('scale-factor')
                plt.ylabel('probability')
                plt.xlim(0, len(plus)+len(minas))
                plt.ylim(-0.1, 1.1)
                plt.hlines(y=0.5, xmin=0, xmax=2*num_augment, colors='r', linestyles='dashed')
                plt.savefig(os.path.join(save_dir, f'frame{img_id}.png'))
                # color = rectangle_color[int(class_id-1)]
                ru.save_result(img_id, args.change_factor, coeffs, Data, PLACE, save_dir)


            elif args.change_factor == 'grid':
                # Data : ('data_point', ('xaxis', 'factor', 'prob'))
                plt.figure(figsize=(12,6))
                for N, index in enumerate(Data.keys()):
                    data = np.array(Data[index])
                    plus = data[data[:,0] == 1]
                    plt.plot(plus[:, 1]*plus[:, 0]+num_augment, plus[:, 2], marker=GRID[PLACE[index].fm], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                            markersize=13, \
                                # label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]'
                                )
                    minas = data[data[:,0] == -1]
                    plt.plot(minas[:, 1]*minas[:,0]+num_augment, minas[:, 2], marker=GRID[PLACE[index].fm], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                            markersize=13, \
                            label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]')
                plt.grid()
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
                plt.subplots_adjust(left=0.1, right=0.8)
                plt.xlabel('grid-factor')
                plt.ylabel('probability')
                plt.xlim(0, len(plus)+len(minas))
                # plt.xlim(0, num_augment*2)
                plt.ylim(-0.1, 1.1)
                plt.hlines(y=0.5, xmin=0, xmax=2*num_augment, colors='r', linestyles='dashed')
                plt.savefig(os.path.join(save_dir, f'frame{img_id}.png'))
                # color = rectangle_color[int(class_id-1)]
                ru.save_result(img_id, args.change_factor, coeffs, Data, PLACE, save_dir)


            elif args.change_factor == 'anchor':
                # # Data : ('data_point', ('xaxis', 'factor', 'prob'))
                # plt.figure(figsize=(12,6))
                # for N, index in enumerate(Data.keys()):
                #     data = np.array(Data[index])
                #     plus = data[data[:,0] == 1]
                #     plt.plot(plus[:, 1]*plus[:, 0], plus[:, 2], marker=GRID[N], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                #             markersize=10, \
                #                 # label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]'
                #                 )
                #     minas = data[data[:,0] == -1]
                #     plt.plot(minas[:, 1]*-1, minas[:, 2], marker=GRID[N], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                #             markersize=10, \
                #             label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]')
                # plt.grid()
                # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
                # plt.subplots_adjust(left=0.1, right=0.8)
                # # plt.title(f'VOC_{img_id}_{args.change_factor}_{coeffs}')
                # plt.xlabel('anchor-factor')
                # plt.ylabel('probability')
                # plt.xlim(-len(minas)-1, len(plus)+1)
                # plt.ylim(-0.1, 1.1)
                # plt.hlines(y=0.5, xmin=-num_augment, xmax=num_augment, colors='r', linestyles='dashed')
                # plt.savefig(os.path.join(save_dir, f'flow_{img_id}-object{b}.png'))
                # # color = rectangle_color[int(class_id-1)]
                # ru.save_result(img_id, args.change_factor, coeffs, Data, PLACE, save_dir)
                # Data : ('data_point', ('xaxis', 'factor', 'prob'))
                plt.figure(figsize=(12,6))
                for N, index in enumerate(Data.keys()):
                    data = np.array(Data[index])
                    plus = data[data[:,0] == 1]
                    plt.plot(plus[:, 1]*plus[:, 0]+num_augment, plus[:, 2], marker=GRID[PLACE[index].fm], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                            markersize=13, \
                                # label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]'
                                )
                    minas = data[data[:,0] == -1]
                    plt.plot(minas[:, 1]*minas[:,0]+num_augment, minas[:, 2], marker=GRID[PLACE[index].fm], color=list(np.array(rectangle_color[N])/255), alpha=0.5,\
                             markersize=13, \
                             label=f'{index}-[{PLACE[index].fm}_{PLACE[index].anchor}_{PLACE[index].grid_y}_{PLACE[index].grid_x}]')
                plt.grid()
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
                plt.subplots_adjust(left=0.1, right=0.8)
                # plt.title(f'VOC_{img_id}_{args.change_factor}_{coeffs}')
                plt.xlabel('anchor-factor')
                plt.ylabel('probability')
                plt.xlim(0, len(plus)+len(minas))
                plt.ylim(-0.1, 1.1)
                plt.hlines(y=0.5, xmin=0, xmax=2*num_augment, colors='r', linestyles='dashed')
                plt.savefig(os.path.join(save_dir, f'frame{img_id}.png'))
                # color = rectangle_color[int(class_id-1)]
                ru.save_result(img_id, args.change_factor, coeffs, Data, PLACE, save_dir)

            else:
                raise Exception('inveld factor')
            print()
            print()


if __name__ == '__main__':
    main()
