#!/bin/bash

# /home/yhosoya/pyfile/paper_implementation/ssd.pytorch/gen_detImages_~~.py 実行sh
conf_thresh=0.50
IoU_thresh=0.50

declare -a file_name=(`ls -d /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/*/`)
declare -a file_name_simple=(`ls /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/`)

# # 要素数の取得
# echo ${#file_name[@]}
# # 配列keyの取得
# echo ${!file_name[@]}
# echo ${!file_name[*]} # @, * どちらでも可

# --image_dir ${file_name[${i}]} \

# train=vgg16_basic
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# for s in grid
# do
#     for i in ${!file_name[@]}
#     do 
#         python survey_one_point_DAVIS_multi.py \
#         --confidence_threshold ${conf_thresh} \
#         --IoU_threshold ${IoU_thresh} \
#         --data_type DAVIS \
#         --image_dir ${file_name[${i}]} \
#         --save_dir ./detect_result/analyse_fragment/${train}_multilabel_thresh02/${s} \
#         --net_type ${net} \
#         --model_weight ${model} \
#         --change_factor ${s} \
#         --fragment_info_dir ./detect_result/find_fragment/${train}_multilabel_thresh02 \
#         --anchors 1,1,1 \
#         --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#         --num_anchors 4,6,6,6,4,4 \
#         --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
#     done
# done



# train=vgg16_sigmoid_thresh_multilabel_thresh02/sw/0406
# net=vgg16-ssd
# model=./models/vgg16_sigmoid_thresh/sw/0406/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth
# for s in grid scale anchor
# do
#     for i in ${!file_name[@]}
#     do 
#         python survey_one_point_DAVIS_multi.py \
#         --confidence_threshold ${conf_thresh} \
#         --IoU_threshold ${IoU_thresh} \
#         --data_type DAVIS \
#         --image_dir ${file_name[${i}]} \
#         --save_dir ./detect_result/analyse_fragment/${train}/${s} \
#         --net_type ${net} \
#         --model_weight ${model} \
#         --change_factor ${s} \
#         --fragment_info_dir ./detect_result/find_fragment/${train} \
#         --anchors 1,1,1 \
#         --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#         --num_anchors 4,6,6,6,4,4 \
#         --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
#     done
# done


# train=vgg16_sigmoid_thresh/ww_0406/run03
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth
# for s in scale grid anchor
# do
#     for i in ${!file_name[@]}
#     do 
#         python survey_one_point_DAVIS.py \
#         --confidence_threshold ${conf_thresh} \
#         --IoU_threshold ${IoU_thresh} \
#         --data_type DAVIS \
#         --image_dir ${file_name[${i}]} \
#         --save_dir ./detect_result/analyse_fragment/${train}/${s} \
#         --net_type ${net} \
#         --model_weight ${model} \
#         --change_factor ${s} \
#         --fragment_info_dir ./detect_result/find_fragment/${train} \
#         --anchors 1,1,1 \
#         --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#         --num_anchors 4,6,6,6,4,4 \
#         --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
#     done
# done


# # train=resnet50_std/multi-step_multilabel_thresh02
# train=resnet50_std_sigmoid/sw/0406/multilabel_thresh02
# net=resnet50-ssd
# # model=./models/resnet50_std/multi-step/resnet50-ssd-Epoch-199-Loss-2.8794355961584275.pth
# model=./models/resnet50_std_sigmoid/sw/0406/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth

# for s in grid scale anchor
# do
#     for i in ${!file_name[@]}
#     do 
#         python survey_one_point_DAVIS_multi.py \
#         --confidence_threshold ${conf_thresh} \
#         --IoU_threshold ${IoU_thresh} \
#         --data_type DAVIS \
#         --image_dir ${file_name[${i}]} \
#         --save_dir ./detect_result/analyse_fragment/${train}/${s} \
#         --net_type ${net} \
#         --model_weight ${model} \
#         --change_factor ${s} \
#         --fragment_info_dir ./detect_result/find_fragment/${train} \
#         --anchors 1,1,1 \
#         --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#         --num_anchors 4,6,6,6,4,4 \
#         --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
#     done
# done



# # compare IoU = binary_0.4
# conf_thresh=0.60
# IoU_thresh=0.50
# train=compare/vgg16_IoU_04
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.989526524082307.pth
# for i in ${!file_name[@]}
# do 
#     for s in anchor
#     do
#         python survey_one_point_DAVIS_multi.py \
#         --confidence_threshold ${conf_thresh} \
#         --IoU_threshold ${IoU_thresh} \
#         --data_type DAVIS \
#         --image_dir ${file_name[${i}]} \
#         --save_dir ./detect_result/analyse_fragment/${train}/${s} \
#         --net_type ${net} \
#         --model_weight ${model} \
#         --change_factor ${s} \
#         --fragment_info_dir ./detect_result/find_fragment/${train} \
#         --anchors 1,1,1 \
#         --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#         --num_anchors 4,6,6,6,4,4 \
#         --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
#     done
# done


# # ==== for slide or thesis image =================================================================
# # vgg16
# conf_thresh=0.6

# # train=vgg16_basic_multilabel_thresh02
# # model=./models/vgg16_basic/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth

# train=vgg16_sigmoid_thresh_multilabel_thresh02/sw/0406
# model=./models/vgg16_sigmoid_thresh/sw/0406/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth

# net=vgg16-ssd

# factor=scale
# # factor=grid
# # factor=anchor

# python survey_one_point_DAVIS_multi.py \
# --confidence_threshold ${conf_thresh} \
# --IoU_threshold ${IoU_thresh} \
# --data_type DAVIS \
# --image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/hike/ \
# --save_dir ./detect_result/analyse_fragment/thesis_image/sigmoid/${factor} \
# --net_type ${net} \
# --model_weight ${model} \
# --change_factor ${factor} \
# --fragment_info_dir ./detect_result/find_fragment/${train} \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512




# resnet50
conf_thresh=0.6

# train=vgg16_basic_multilabel_thresh02
# model=./models/vgg16_basic/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth

train=resnet50_std_sigmoid/sw/0406/multilabel_thresh02
model=./models/resnet50_std_sigmoid/sw/0406/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth

net=resnet50-ssd

# factor=scale
factor=grid
# factor=anchor

python survey_one_point_DAVIS_multi.py \
--confidence_threshold ${conf_thresh} \
--IoU_threshold ${IoU_thresh} \
--data_type DAVIS \
--image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/mbike-trick/ \
--save_dir ./detect_result/analyse_fragment/thesis_image/resnet50/sigmoid/${factor} \
--net_type ${net} \
--model_weight ${model} \
--change_factor ${factor} \
--fragment_info_dir ./detect_result/find_fragment/${train} \
--anchors 1,1,1 \
--rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
--num_anchors 4,6,6,6,4,4 \
--vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512