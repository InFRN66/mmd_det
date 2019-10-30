import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.resnet50_ssd import (create_resnet50_ssd, create_resnet50_ssd_v2, 
                                     create_resnet50_ssd_predictor, convert_key, modifie_weight)
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from vision.ssd.config.vgg_ssd_config import Config as Config_vgg
from vision.ssd.config.resnet50_ssd_config import Config as Config_resnet
import argparse
import pathlib
import numpy as np
import logging
import sys, os
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

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


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, default="./models/voc-model-labels.txt", help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--mAP_iou_threshold", type=float, default=0.5, help="The threshold for mAP measurement.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

parser.add_argument('--anchors', type=to_list01,
                    help='selection useing anchor [small sq, big sq, rects]', required=False)        
parser.add_argument('--rectangles', action="append", nargs="+", type=int, 
                    help='rectangles to use', required=False)
parser.add_argument('--num_anchors', type=to_list01, help='num anchors in each fm', required=False) 
parser.add_argument('--vgg_config', type=to_list02, help='vgg network configuration', required=False)
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

print(args.use_cuda, torch.cuda.is_available())
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")
os.makedirs(args.eval_dir, exist_ok=True)


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i) # len(annotation) = 3 , (coords[num_target, 4], labels[num_target], is_difficult[num_target])
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id]) # [1, 4]
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases # each class, each image(id)の情報


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):

    """
    num_true_cases : 実際にそのクラスに属するtarget総数
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f: # そのクラス分のprediction box 全部読み込み
            t = line.rstrip().split(" ") # t: [image_id, scores, boxes]
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores) # -付きなのでlarger is firstでindex // 全ボックス内でconfidenceの高い順にsort
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids): # image_ids : このクラスと判断されたimageのid / gt_boxes : 実際にこのクラスだったボックス
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1 # このクラスの物体box，と予測して実際は違った
                continue

            gt_box = gt_boxes[image_id] # 多分複数ボックス（そのimage_id中のこのクラスの正解ボックス）
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold: # 少なくともどれかのgtboxに対してIoU>しきい値であったpredictionが存在
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched: # これまで，同じimage_id中で今見てるボックス（max_arg）が検出されていない場合(not in match)
                        true_positive[i] = 1 # max_argのボックス = true positive
                        matched.add((image_id, max_arg))
                    else: # 同じgt_boxが既に検出され，true positiveとして数えられている:　同じ物体を重複してpredicしている場合など．
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    if args.net == "vgg16-ssd":    
        config = Config_vgg(anchors=args.anchors, rectangles=args.rectangles, 
                            num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None) # --- custom
    elif args.net == "resnet50-ssd":
        config = Config_resnet(anchors=args.anchors, rectangles=args.rectangles,
                               num_anchors=args.num_anchors, vgg_config=args.vgg_config, pretrained=None)  # --- custom


    print("+++++++++++++++++++++++")
    print(config.image_mean, config.image_std)
    print()

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), config, is_test=True)
    elif args.net == 'resnet50-ssd':
        # net = create_resnet50_ssd(len(class_names), config, is_test=True)
        net = create_resnet50_ssd_v2(len(class_names), config, is_test=True)
    # elif args.net == 'mb1-ssd':
    #     net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    # elif args.net == 'mb1-ssd-lite':
    #     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    # elif args.net == 'sq-ssd-lite':
    #     net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    # elif args.net == 'mb2-ssd-lite':
    #     net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    # --- load for resnet 79.7 model
    if args.trained_model.split("/")[-1] == "resnet50_ssd_voc_79.7.pth" and args.net == "resnet50-ssd":
        print("Load resnet 79.7 model")
        pretrained_dict = torch.load(args.trained_model)
        net_dict = net.state_dict() # base_net, num_batch...
        pretrained_dict = convert_key(pretrained_dict)
        # # 1. fliter \ netにないkeyは消す
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        # # 2. overwrite
        net_dict.update(pretrained_dict)
        # # 3. load
        net.load_state_dict(net_dict)
    else:
        net.load(args.trained_model)
        
    
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE, config=config) # iou_theresh : nmsのしきい値
    elif args.net == 'resnet50-ssd':
        predictor = create_resnet50_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE, config=config)
    # elif args.net == 'mb1-ssd':
    #     predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    # elif args.net == 'mb1-ssd-lite':
    #     predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    # elif args.net == 'sq-ssd-lite':
    #     predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    # elif args.net == 'mb2-ssd-lite':
    #     predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i) # original [h,w,c]

        # # RGB 2 BGR
        # image = image[:,:,::-1]

        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image) # [num_box, 4], [num_box], [num_box] // nms後のもの. prob_thrshold=..で無視するボックス設定 

        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i # [num_bb, 1], value=i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1)) # results[i] = [num_box, 7]in columns, [index, label, prob, boxes+1]
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :] # class indexに該当する箇所だけ選択
            for i in range(sub.size(0)): # num_boxes in  this class
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    # with open(os.path.join(args.eval_dir, 'all.txt'), 'w') as f:
    f = open(os.path.join(args.eval_dir, 'all.txt'), 'w')
    
    aps = []
    print("\n\nAverage Precision Per-class:")
    f.write("Average Precision Per-class:\n\n")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.mAP_iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")
        f.write(f"{class_name}: {ap}\n")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    f.write(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")


