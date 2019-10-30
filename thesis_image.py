"""
Do ordinary object detection, generate detected images. Thats all.
Detect objects, depict rectangles and save them.
sh :: [do_detect.sh]
"""
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.resnet50_ssd import create_resnet50_ssd_predictor, create_resnet50_ssd_v2
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.config.vgg_ssd_config import Config as Config_vgg
from vision.ssd.config.resnet50_ssd_config import Config as Config_resnet
from vision.utils.misc import Timer
import torch
import cv2
from PIL import Image
import numpy as np
import sys, os, argparse, glob, pickle


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
parser.add_argument("--model_weight", type=str, default="models/vgg16-ssd-mp-0_7726.pth", 
                    help="model weight path to use")

parser.add_argument('--anchors', type=to_list01, help='selection useing anchor [small sq, big sq, rects]',
                    required=True)        
parser.add_argument('--rectangles', action="append", nargs="+", type=int, 
                    help='rectangles to use', required=True) 
parser.add_argument('--num_anchors', type=to_list01, help='num anchors in each fm', required=True) 
parser.add_argument('--vgg_config', type=to_list02, help='vgg network configuration', required=True) 
args = parser.parse_args()

net_type = args.net_type
print("Net_Typoe: ", args.net_type)
print("Model_Weight: ", args.model_weight)
model_path = args.model_weight
label_path = args.label_path

# image_path = '~/pyfile/paper_implementation/ssd_visialize/test_image/dog.jpg'
class_names = [name.strip() for name in open(label_path).readlines()]
rectangle_color = pickle.load(open("pallete", "rb"))

def prepare_save_and_images():
    if args.save_dir is None:
        save_dir = args.image_dir
        save_dir = os.path.join(args.image_dir, args.file_Name, str(args.net_type)+'_'+str(args.confidence_threshold))
    else:
        save_dir = os.path.join(args.save_dir, args.image_dir.split('/')[-2])
    os.makedirs(save_dir, exist_ok=True)
    
    image_dir = args.image_dir
    img_list = glob.glob(image_dir + "/*.jpg") # all images in dir
    img_list.extend(glob.glob(image_dir + "/*.JPEG"))
    img_list.sort()
    
    print("loaded {} images".format(len(img_list)))

    # annotation load
    # pix, obj = args.image_dir.split('/')[-3:-1]
    # paths = args.image_dir.split('/')[1:-4]
    # paths.extend(['Annotations', pix, obj])
    # annot_dir = os.path.join(*paths)
    # annot_dir = '/'+annot_dir

    # with open(os.path.join(annot_dir,'label.json'), 'r') as f:
    #     annotation = json.load(f)

    org_images = []
    for j in range(len(img_list)):
        imgpath = img_list[j]
        org_im = cv2.imread(imgpath)
        org_images.append(cv2.cvtColor(org_im, cv2.COLOR_BGR2RGB)) 
 
    return save_dir, img_list, org_images  # RGB



def load_model(net_type):
    # if args.net == "vgg16-ssd":
    #     config = Config_vgg(anchors=args.anchors, rectangles=args.rectangles,
    #                         num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None)  # --- custom
    # elif args.net == "resnet50-ssd":
    #     config = Config_resnet(anchors=args.anchors, rectangles=args.rectangles,
    #                            num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None)  # --- custom
    # # config = Config(anchors=args.anchors, rectangles=args.rectangles, 
    # #                 num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None) # --- custom
    if net_type == 'vgg16-ssd':
        # net = create_vgg_ssd(len(class_names), is_test=True)
        config = Config_vgg(anchors=args.anchors, rectangles=args.rectangles,
                            num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None)  # --- custom
        net = create_vgg_ssd(len(class_names), config, is_test=True)
    elif net_type == 'resnet50-ssd':
        config = Config_resnet(anchors=args.anchors, rectangles=args.rectangles,
                               num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None)  # --- custom
        net = create_resnet50_ssd_v2(len(class_names), config, is_test=True)
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
        predictor = create_vgg_ssd_predictor(net, candidate_size=200, config=config)
    elif net_type == 'resnet50-ssd':
        predictor = create_resnet50_ssd_predictor(net, candidate_size=200, config=config)
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


def detection_rectangle_write(image, dets, idx, class_id):
    """
    detection boxを描画．
    image : [h,w,c], numpy, RGB
    dets : [num_bb, 5] conf,x1,y1,x2,y2 in one class, one image
    class_id : 1 ~ 80 or 1 ~ 20, because 0 is background and to be skipped
    """
    if len(dets) == 0:
        # print("no detection was found.")
        return image

    for i in range(len(dets)):
        score = dets[i, 0]
        x1 = dets[i, 1]
        y1 = dets[i, 2]
        x2 = dets[i, 3]
        y2 = dets[i, 4]
        color = rectangle_color[int(class_id - 1)]
        print(class_id)

        color = (255, 0, 0)

        # if class_id == 15:
        #     color = (255, 0, 0)
        # else:
        #     continue
        # elif class_id == 2:
        #     color = (128, 0, 255)
        if class_id != 15:
            continue

        if i != 1:
            continue

        label = class_names[int(class_id)] + ':' + str(round(score.item(), 4))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        # cv2.putText(image, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1.7, [0,0,0], 3) # 大きさ，太さ
        # cv2.putText(image, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1.7, [255,255,255], 1) # 大きさ，太さ
        # break

    return image


def main():
    # make save dir, load org_img, or etc 
    save_dir, img_list, org_images = prepare_save_and_images()
    # load net 
    predictor = load_model(net_type)
    for idx in range(len(img_list)):

        written_image = None
        # boxes, labels, probs = predictor.predict(org_images[idx], top_k=10, prob_threshold=args.confidence_threshold)
        detections, selected_indicies, height, width = predictor.predict_for_genImage(org_images[idx], top_k=10, 
                                                                                      prob_threshold=args.confidence_threshold)
        # detections : [1, 21, top_k, 5]
        # selected_indicies : [1, 21, top_k]     
        
        for c in range(1, 21):
            row_dets = detections[0, c, :] # [top_k, 5]
            conf_mask = row_dets[:,0].gt(args.confidence_threshold) # [top_k, 5]
            conf_mask = conf_mask.view(-1,1).expand_as(row_dets) # [top_k, 5]
            dets = torch.masked_select(row_dets, conf_mask).view(-1, 5)
            if dets.shape == torch.Size([0]): # detect結果無しの場合
                continue

            dets[:,1] *= width
            dets[:,2] *= height
            dets[:,3] *= width
            dets[:,4] *= height
            print("detes", dets.shape)

            if written_image is None: # 初のクラス書き込み
                written_image = detection_rectangle_write(org_images[idx], dets, idx, c)
                # written_image = detection_rectangle_write(rgb_image, new_dets, idx, c, frame, found)
            else:
                written_image = detection_rectangle_write(written_image, dets, idx, c)
                # written_image = detection_rectangle_write(written_image, new_dets, idx, c, frame, found)
        
        if written_image is None:
            written_image = org_images[idx]
            print("writte image is None.")

        path = os.path.join(save_dir, img_list[idx].split('/')[-1])
        print("PATH", path)
        # --- save image
        # cv2.imwrite(path, written_image)
        Image.fromarray(np.uint8(written_image)).save(path)
        
        # for i in range(boxes.size(0)):
        #     box = boxes[i, :]
        #     cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #     #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        #     # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        #     cv2.putText(orig_image, label,
        #                 (box[0] + 20, box[1] + 40),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1,  # font scale
        #                 (255, 0, 255),
        #                 2)  # line type
        #     # path = "run_ssd_example_output.jpg"
        #     path = os.path.join(save_dir, img_list[i])
        #     cv2.imwrite(path, orig_image)
        #     # print(f"Found {len(probs)} objects. The output image is {path}")


if __name__ == '__main__':
    main()
