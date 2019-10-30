# --- find fragment frames

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.resnet50_ssd import create_resnet50_ssd, create_resnet50_ssd_predictor
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.config.vgg_ssd_config import Config
from vision.utils.misc import Timer
import torch
import cv2
from PIL import Image
import numpy as np
import sys, os, argparse, glob, pickle, json
from pathlib import Path
# sys.path.append('../ssd_visualize')
# from flicker_oneObject_mean import Frame
from flicker_oneObject import Frame


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
parser.add_argument("--track_Id", type=int, default=2, help="class id to track (object seen in images)")
parser.add_argument("--save_dir", type=str, default=None, help="dir to save det_images")
parser.add_argument("--image_dir", type=str, default="images/", help="dir to images")
parser.add_argument("--file_Name", type=str, default="ssd_moblieNet", help="save_folder")

parser.add_argument("--confidence_threshold", default=0.5, type=float, help='Detection confidence threshold')
parser.add_argument("--IoU_threshold", default=0.5, type=float, help='Minimum IoU threshold for checking the same object')
parser.add_argument("--net_type", default="vgg16-ssd", type=str, help='model to use')
parser.add_argument("--label_path", default='models/voc-model-labels.txt', type=str, 
                    help='path to VOC label')
parser.add_argument("--model_weight", type=str, default="models/trained/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth", 
                    help="model weight path to use")

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

config = Config(args.anchors, args.rectangles, args.num_anchors, args.vgg_config)

target_name = args.image_dir.split('/')[-2] if args.image_dir.endswith('/') else args.image_dir.split('/')[-1]
print("target: ", target_name)

class_names = [name.strip() for name in open(label_path).readlines()]
rectangle_color = pickle.load(open("pallete", "rb"))

# hyper params
num_track_objects = len([int(args.track_Id)])
num_classes = len(class_names)
top_k = 200

def prepare_save_and_images():
    if args.save_dir is None:
        save_dir = args.image_dir
        save_dir = os.path.join(args.image_dir, args.file_Name, str(args.net_type)+'_'+str(args.confidence_threshold))
    else:
        save_dir = os.path.join(args.save_dir, target_name)
    os.makedirs(save_dir, exist_ok=True)
    
    image_dir = args.image_dir
    img_list = glob.glob(image_dir + "/*.jpg") # all images in dir
    img_list.extend(glob.glob(image_dir + "/*.JPEG"))
    img_list.sort()
    
    print("loaded {} images".format(len(img_list)))

    # annotation load
    pix, obj = args.image_dir.split('/')[-3:-1]
    paths = args.image_dir.split('/')[1:-4]
    paths.extend(['Annotations', pix, obj])
    annot_dir = os.path.join(*paths)
    annot_dir = '/'+annot_dir
    with open(os.path.join(annot_dir,'label.json'), 'r') as f:
        annotation = json.load(f)

    # make list for original_images(RGB, [h,w,c])
    org_images = []
    for j in range(len(img_list)):
        imgpath = img_list[j]
        org_im = cv2.imread(imgpath)
        org_images.append(cv2.cvtColor(org_im, cv2.COLOR_BGR2RGB)) 

    gt_boxes = list()
    for i in range(len(annotation)-2): # h, wは除く
        box = annotation[str(i)]
        box = np.array([box['x1'], box['y1'], box['x2'], box['y2']])
        gt_boxes.append(box)
    gt_boxes = np.vstack(gt_boxes)

    return save_dir, img_list, org_images, annot_dir, annotation, gt_boxes  # RGB



def load_model(net_type):
    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), config, is_test=True)
        # net = create_vgg_ssd_custom(len(class_names), anchor_status=[3,3,3,3,3,3], is_test=True) # ------------------------------------------------------
    elif net_type == 'resnet50-ssd':
        net = create_resnet50_ssd(len(class_names), config, is_test=True)
    # elif net_type == 'mb1-ssd':
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
        predictor = create_vgg_ssd_predictor(net, config=config, candidate_size=200)
    elif net_type == 'resnet50-ssd':
        predictor = create_resnet50_ssd_predictor(net, config=config, candidate_size=200)
    # elif net_type == 'mb1-ssd':
    #     predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    # elif net_type == 'mb1-ssd-lite':
    #     predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    # elif net_type == 'mb2-ssd-lite':
    #     predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    # elif net_type == 'sq-ssd-lite':
    #     predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    
    return predictor


def detection_rectangle_write(image, dets, idx, class_id, found):
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
        color = rectangle_color[int(class_id-1)]
        label = class_names[int(class_id)] + ':' + str(round(score.item(), 4))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1.7, [0,0,0], 3) # 大きさ，太さ
        cv2.putText(image, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1.7, [255,255,255], 1) # 大きさ，太さ

    return image


def main():
    # make save dir, load org_img, or etc 
    save_dir, img_list, org_images, annot_dir, annotation, gt_boxes = prepare_save_and_images()
    # load net 
    predictor = load_model(net_type)
    
    # frame instance construct フレーム管理オブジェクト
    frame = Frame(num_scale=len(img_list), num_classes=num_classes, num_bb=num_track_objects, top_k=top_k, \
                    h=org_images[0].shape[0], w=org_images[0].shape[1], c=org_images[0].shape[2])

    selects = np.zeros(len(img_list))

    for idx in range(len(img_list)):
        written_image = None
        # boxes, labels, probs = predictor.predict(org_images[idx], top_k=10, prob_threshold=args.confidence_threshold)
        detections, selected_indicies, height, width = predictor.predict_for_genImage(org_images[idx], top_k=top_k, 
                                                                                      prob_threshold=args.confidence_threshold)
        # detections : [1, 21, top_k, 5]
        # selected_indicies : [1, 21, top_k]
        print(f'frame {idx}')
        print('Selected:', selected_indicies[0, args.track_Id, :5])
        print('dets: ', detections[0, args.track_Id, :5, 0])
        #  for c in range(1, 21):
        for c in [args.track_Id]:
            row_dets = detections[0, c, :] # [top_k, 5]
            
            conf_thresh = args.confidence_threshold
            IoU_thresh = args.IoU_threshold
            ret = 0

            while ret == 0:
                conf_mask = row_dets[:,0].gt(conf_thresh) # [top_k, 5]
                conf_mask = conf_mask.view(-1,1).expand_as(row_dets) # [top_k, 5]
                
                dets = torch.masked_select(row_dets, conf_mask).view(-1, 5)
                if dets.shape == torch.Size([0]): # detect結果無しの場合
                    ret, found, sidx = frame.no_detection(idx, c, 0)
                    conf_thresh = conf_thresh*0.60
                    IoU_thresh = IoU_thresh*0.60

                    if conf_thresh <= 0.1 and ret == 0:
                        print("thresh limit : can't detect any box.")
                        ret = 1
                        frame.track_box[idx, c, 0, 1:] = frame.track_box[idx-1, c, 0, 1:] # 前フレームと同じにする
                        break
                    continue
            
                dets[:,1] *= width
                dets[:,2] *= height
                dets[:,3] *= width
                dets[:,4] *= height

                frame.class_act(idx, c, 0, dets) # 取り敢えずoutputを保存
                ret, found, sidx = frame.compare_IoU_one(idx, c, 0, dets, gt_boxes[[idx], :], IoU_thresh=IoU_thresh)

                # 見つからなかったら徐々にしきい値を下げる
                conf_thresh = conf_thresh*0.60
                IoU_thresh = IoU_thresh*0.60
                
                if conf_thresh <= 0.1:
                    print("thresh limit : can't detect any box.")
                    ret = 1
                    frame.track_box[idx, c, 0, 1:] = frame.track_box[idx-1, c, 0, 1:]
                    
            print('current_status :', frame.trajectory[idx].item())
            print('sidx', sidx)
            # 実際に追跡するbox
            new_dets = frame.track_box[idx, c, [0], :] # [1,5] prob, coord

            P = new_dets[0, 0] # trackするボックスのprob

            if P == 0:
                selects[idx] = selects[idx-1] # 前フレームと同じにする
            else:
                selects[idx] = selected_indicies[0, args.track_Id, np.where(row_dets[:, 0]==P)[0]]

            print('new_dets, ',new_dets.shape, new_dets, selects)

            if written_image is None: # 初のクラス書き込み
                # written_image = detection_rectangle_write(org_images[idx], dets, idx, c)
                written_image = detection_rectangle_write(org_images[idx], new_dets, idx, c, found)
            else:
                # written_image = detection_rectangle_write(written_image, dets, idx, c)
                written_image = detection_rectangle_write(written_image, new_dets, idx, c, found)
        
        if written_image is None:
            written_image = org_images[idx]
            print("written image is None.")

        frame.frame_act(idx, written_image)
        print()

    # print('TTT', selects)
    # set1 : 前後のprobの差がprob_threshより大きいものを，flickerとして抽出(長く低確率が続くものを含めないようにしている)
    # frame status survey
    # frame_status = frame.analyse(args.track_Id)

    # set2 : filter0.9(トラジェクトリの中で一回でも0.9を超えた場合：そのトラジェクトリ内でconf_threshを下回ったものを全て抽出)
    # frame_status = frame.filter_highprob_trajectory(args.track_Id, args.confidence_threshold)
    # frame_status = frame.analyse_occlusion(args.track_Id, frame_status)

    # set3 : 多分high_pass前フレームのprobを100として，何％ptrobが下がったか
    frame_status = frame.trajectory.cpu().numpy().reshape((-1))
    print('final before status ', frame_status.shape)
    prob_status, probability = frame.analyse_probabiity_percent(args.track_Id, frame_status, obj=0)
    frame_status = frame.analyse(prob_status, frame_status, args.track_Id, prob_thresh=0.4)  # 40%下がったら検出
    

    frame_status = frame.trajectory.cpu().numpy().reshape((-1))
    print("frame_status")
    








    # # ------------------------------ new method: 
    # trajectory = frame.trajectory.cpu().numpy()[:, 0].reshape((-1)) # 0 or 1 // 1でmiss
    # print(f'trajectory:\n {trajectory}')
    
    # trajectory *= frame.box_hide_status[:, 0].reshape(-1)
    # print(f'trajectory with box restrict:\n {trajectory}')

    # fragment_candidate = frame.fragment_candidate(trajectory)
    # print(f'fragment_candidate:\n {fragment_candidate}')
    
    # fragment_result = frame.use_candidate_and_trajectory(trajectory, fragment_candidate)
    # print(f'fragment_result:\n {fragment_result}')

    # probability = frame.track_box[:, c, 0, 0].cpu().numpy()
    # print(f'probability:\n {probability}')
    # # ------------------------------------------------


    for f, s in enumerate(frame.trajectory.cpu().numpy()):
        if s == 1:
            s = 2 # 純粋にprobが低いフレームは取り敢えず緑をつけておく
            name = img_list[f].split("/")[-1]
            frame.add_flickSign(f, s)


    # print(frame_status.shape, frame_status)
    # for f, s in enumerate(frame_status):
    #     # track
    #     if s == 0:
    #         name = img_list[f].split("/")[-1]
    #         # frame.add_flickSign(f, s)
    #         Image.fromarray(np.uint8(frame.written_image[f])).save(os.path.join(save_dir, str(name)))
    #         print("saved {}".format(str(name)))
    #     # flicker
    #     elif s == 1:
    #         name = img_list[f].split("/")[-1]
    #         frame.add_flickSign(f, s)
    #         Image.fromarray(np.uint8(frame.written_image[f])).save(os.path.join(save_dir, str(name)))
    #         print("saved {}".format(str(name)))
    #     # occlusion
    #     elif s == -1:
    #         name = img_list[f].split("/")[-1]
    #         frame.add_flickSign(f, s)
    #         Image.fromarray(np.uint8(frame.written_image[f])).save(os.path.join(save_dir, str(name)))
    #         print("saved {}".format(str(name)))
    #     else:
    #         raise Exception('invalid status {}.'.format(s))


    num_flicker = (frame_status==1).sum()
    print('Conf_thresh:{},  flicker rate :{}/{} [{}]'.format(args.confidence_threshold, num_flicker, \
                                                             frame.num_scale, round(num_flicker/(frame.num_scale+1e-8), 4)))
    # use self-train weights
    log_dir = Path(args.save_dir)

    # name = args.image_dir.split('/')[-2] if len(args.image_dir.split('/')[-1])==0 else args.image_dir.split('/')[-1]
    name = Path(args.image_dir).name
    if os.path.isfile(os.path.join(log_dir, 'DAVIS_flicker.json')):
        with open(os.path.join(log_dir, 'DAVIS_flicker.json'), 'r') as f1:
            o = json.load(f1)
    else:
        o = dict()
    
    o[name] = dict()
    o[name]['conf_thresh'] = args.confidence_threshold
    o[name]['IoU_thresh'] = args.IoU_threshold
    o[name]['num_frame'] = frame.num_scale
    o[name]['num_flicker'] = float(num_flicker)
    o[name]['flicker_rate'] = round(num_flicker/(frame.num_scale+1e-8), 4)
    o[name]['miss'] = frame.trajectory.cpu().numpy().tolist()
    o[name]['flicker_indicies'] = frame_status.tolist()
    o[name]['selected_indicies'] = selects.tolist()
    o[name]['track_Id'] = args.track_Id
    o[name]['probability'] = probability.tolist()
    o[name]["full_image_list"] = [ img_list[i] for i in range(len(frame_status)) if frame_status[i] == 1 ]
    o[name]["image_list"] = [ x.split("/")[-1] for x in o[name]["full_image_list"] ]

    with open(os.path.join(log_dir, 'DAVIS_flicker.json'), 'w') as f2:
        json.dump(o, f2)


    # # ---- version 2 ?
    # for f, s in enumerate(trajectory):
    #     if s == 1:
    #         s = 2 # 純粋にprobが低いフレームは取り敢えず緑をつけておく
    #         name = img_list[f].split("/")[-1]
    #         frame.add_flickSign(f, s)

    # for f, s in enumerate(fragment_result):
    #     name = img_list[f].split("/")[-1]
    #     os.makedirs(os.path.join(save_dir), exist_ok=True)
    #     # track
    #     if s == 0:
    #         Image.fromarray(np.uint8(frame.written_image[f])).save(os.path.join(save_dir, str(name)))
    #         # print("saved {}".format(str(name)))
    #     # flicker
    #     elif s == 1:
    #         frame.add_flickSign(f, s)
    #         Image.fromarray(np.uint8(frame.written_image[f])).save(os.path.join(save_dir, str(name)))
    #         # print("saved {}".format(str(name)))
    #     else:
    #         raise Exception('invalid status {}.'.format(s))

    # # print('MISS:')
    # # print(frame.trajectory.cpu().numpy())

    # num_flicker = (fragment_result==1).sum()
    # print(num_flicker)
    # print()
    # print('Conf_thresh:{},  flicker rate :{}/{} [{}]'.format(args.confidence_threshold, num_flicker, \
    #                                                         frame.num_scale, round(num_flicker/(frame.num_scale+1e-8), 4)))
    # # use self-train weights
    # log_dir = os.path.join(save_dir)
    # os.makedirs(log_dir, exist_ok=True)

    # if os.path.isfile(os.path.join(log_dir, 'DAVIS_flicker.json')):
    #     with open(os.path.join(log_dir, 'DAVIS_flicker.json'), 'r') as f1:
    #         o = json.load(f1)
    # else:
    #     o = dict()
    
    # o['conf_thresh'] = args.confidence_threshold
    # o['IoU_thresh'] = args.IoU_threshold
    # o['num_frame'] = frame.num_scale
    # o['num_flicker'] = float(num_flicker)
    # o['flicker_rate'] = round(num_flicker/(frame.num_scale+1e-8), 4)
    # o['miss'] = trajectory.tolist()
    # o['flicker_indicies'] = fragment_result.tolist()
    # o['selected_indicies'] = selects.tolist()
    # o['track_Id'] = str(c)
    # o['probability'] = probability.tolist()

    # with open(os.path.join(log_dir, 'DAVIS_flicker.json'), 'w') as f2:
    #     json.dump(o, f2)
    # # ---- version 2 ?

if __name__ == '__main__':
    main()




    