#!/bin/bash
# model=./models/resnet50/resnet50-ssd-Epoch-199-Loss-2.105243469053699.pth
# model=./train_info/my_train/resnet50-ssd/resnet50-ssd-Epoch-199-Loss-3.0232349241933516.pth # --- trained in RGB



# # --- for vgg16-ssd + fractional pooling w/o (conv + relu) 
# net=vgg16-ssd
# model=./models/vgg16_fracP_woConvReLU/vgg16-ssd-Epoch-199-Loss-2.9701730282075944.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,"F11","F12",128,128,"F21","F22",256,256,256,"F31","F32",512,512,512,"F41","F42",512,512,512 \
# --eval_dir ./models/vgg16_fracP_woConvReLU/eval


# # --- for vgg16-ssd - sigmoid
# train=sw/0307
# net=vgg16-ssd
# model=./models/vgg16_sigmoid_thresh/${train}/vgg16-ssd-Epoch-200-Loss-2.8186199772742486.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --eval_dir ./models/vgg16_sigmoid_thresh/${train}/eval


# trainPath=(
#     sw/0305
#     sw/0306
#     sw/0307
#     sw/0405
#     sw/0406
#     sw/035055
#     sw/03506
#     sw/035065
#     sw/04055
#     sw/045055
# )
# Models=(
#     vgg16-ssd-Epoch-200-Loss-2.7386556971457696.pth
#     vgg16-ssd-Epoch-200-Loss-2.773837577143023.pth
#     vgg16-ssd-Epoch-200-Loss-2.8186199772742486.pth
#     vgg16-ssd-Epoch-200-Loss-2.8535977763514366.pth
#     vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth
#     vgg16-ssd-Epoch-200-Loss-2.801233918436112.pth
#     vgg16-ssd-Epoch-200-Loss-2.759871079075721.pth
#     vgg16-ssd-Epoch-200-Loss-2.7583961202252296.pth
#     vgg16-ssd-Epoch-200-Loss-2.7950822553327006.pth
#     vgg16-ssd-Epoch-200-Loss-2.6780950776992305.pth    
# )
# net=vgg16-ssd
# for j in ${!trainPath[@]}
# do
    
#     train=${trainPath[${j}]}
#     net=vgg16-ssd
#     model=./models/vgg16_sigmoid_thresh/${trainPath[${j}]}/${Models[${j}]}
#     python eval_ssd.py \
#     --net ${net} \
#     --trained_model ${model} \
#     --dataset_type voc \
#     --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
#     --eval_dir ./models/vgg16_sigmoid_thresh/${train}/eval
# done


# net=resnet50-ssd
# method=multi-step
# model=./models/resnet50_std/${method}/resnet50-ssd-Epoch-199-Loss-2.8794355961584275.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --eval_dir ./models/resnet50_std/${method}/eval

# net=resnet50-ssd
# method=resnet50_std_sigmoid/sw/0406
# # model=./models/resnet50_std/${method}/resnet50-ssd-Epoch-199-Loss-2.8794355961584275.pth
# model=./models/${method}/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --eval_dir ./models/${method}/eval


# # vgg16 complement with fragment
# net=vgg16-ssd
# method=complement_with_fractional/vgg16_basic
# model=./models/${method}/vgg16-ssd-Epoch-199-Loss-2.923230706491778.pth
# python eval_ssd_complement_with_fractional.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --num_anchors 4,6,6,6,4,4 \
# --eval_dir ./models/${method}/eval


# # vgg16 matching IoU = 0.5, basic
# net=vgg16-ssd
# method=vgg16_basic
# model=./models/${method}/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --num_anchors 4,6,6,6,4,4 \
# --eval_dir ./models/${method}/eval


# vgg16 matching IoU = 0.4
net=vgg16-ssd
method=compare/vgg16_IoU_04
# model=./models/${method}/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
model=../ssd_train/train_info/matching_IoU/vgg16-ssd/0.4/vgg16-ssd-Epoch-199-Loss-2.997551424272599.pth
python eval_ssd.py \
--net ${net} \
--trained_model ${model} \
--dataset_type voc \
--dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
--anchors 1,1,1 \
--rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
--vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
--num_anchors 4,6,6,6,4,4 \
--eval_dir ./models/${method}_v2/eval \
--mAP_iou_threshold 0.5

# method=compare/vgg16_IoU_04
# model=./models/${method}/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth