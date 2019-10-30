#!/bin/bash

conf_thresh=0.6
IoU_thresh=0.6

# # 要素数の取得
# echo ${#file_name[@]}

# # 配列keyの取得
# echo ${!file_name[@]}
# echo ${!file_name[*]} # @, * どちらでも可


# declare -a target_class=(2 3 15 15 4 15 15 15 6 7 7 7 15 7 10 15 15 15 15 15 12 12 12 15 12 7 7 7 3 15 15 13 13 15 \
#                          15 15 15 15 15 15 12 15 15 15 15 3 3 14 14 14 14 7 15 15 1 7 15 14 15 14 17 15 15 15 15 15 \
#                          4 15 15 7 19 15 15)

declare -a file_name=(`ls -d /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/*/`)
# declare -a file_name=(`ls -d /mnt/disk1/img_data/VOC_random_seed/scale_1.01/seed4/*/`)


# # vgg16 - basic model matching IoU = 0.5
# train=vgg16_basic_multilabel_thresh04
# net=vgg16-ssd
# model=./models/vgg16_basic/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --save_dir ./detect_result/find_fragment/${train} \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
# done


# vgg16 - basic model matching IoU = 0.4
train=compare/vgg16_IoU_04
net=vgg16-ssd
model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.989526524082307.pth
for i in ${!file_name[@]}
do
    python find_fragment.py \
    --data_type voc \
    --save_dir ./detect_result/find_fragment/${train} \
    --image_dir ${file_name[${i}]} \
    --IoU_threshold ${IoU_thresh} \
    --confidence_threshold ${conf_thresh} \
    --net_type ${net} \
    --model_weight ${model} \
    --anchors 1,1,1 \
    --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
    --num_anchors 4,6,6,6,4,4 \
    --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
done


# # vgg16 - basic model complement with fractional pooling
# train=complement_with_fractional/vgg16_basic
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.923230706491778.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --save_dir ./detect_result/find_fragment/${train} \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
#     --complement_fractional
# done


# # --- for vgg16-ssd + fractional pooling w/o (conv + relu) 
# train=vgg16_base
# net=vgg16-ssd
# model=./models/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --track_Id ${target_class[${i}]} \
#     --save_dir ./detect_result/find_fragment/${train} \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
# done


# # --- for vgg16-ssd + fractional pooling w/o (conv + relu) 
# train=vgg16_fracP_woConvReLU
# net=vgg16-ssd
# model=./models/vgg16_fracP_woConvReLU/vgg16-ssd-Epoch-199-Loss-2.9701730282075944.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --track_Id ${target_class[${i}]} \
#     --save_dir ./detect_result/find_fragment/${train} \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,"F11","F12",128,128,"F21","F22",256,256,256,"F31","F32",512,512,512,"F41","F42",512,512,512
# done


# # --- for vgg16-ssd sigmoid sampling 
# train=sw/0406
# net=vgg16-ssd
# model=./models/vgg16_sigmoid_thresh/${train}/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --save_dir ./detect_result/find_fragment/vgg16_sigmoid_thresh_multilabel_thresh02/${train} \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
# done


#trainPath=(
#    # sw/0305
#    sw/0306
#    # sw/0307
#    sw/0405
#    # sw/0406
#    # sw/035055
#    # sw/03506
#    # sw/035065
#    # sw/04055
#    # sw/045055
#)
#Models=(
#    # vgg16-ssd-Epoch-200-Loss-2.7386556971457696.pth
#    vgg16-ssd-Epoch-200-Loss-2.773837577143023.pth
#    # vgg16-ssd-Epoch-200-Loss-2.8186199772742486.pth
#    vgg16-ssd-Epoch-200-Loss-2.8535977763514366.pth
#    # vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth
#    # vgg16-ssd-Epoch-200-Loss-2.801233918436112.pth
#    # vgg16-ssd-Epoch-200-Loss-2.759871079075721.pth
#    # vgg16-ssd-Epoch-200-Loss-2.7583961202252296.pth
#    # vgg16-ssd-Epoch-200-Loss-2.7950822553327006.pth
#    # vgg16-ssd-Epoch-200-Loss-2.6780950776992305.pth    
#)
#net=vgg16-ssd
#for j in ${!trainPath[@]}
#do
#    model=./models/vgg16_sigmoid_thresh/${trainPath[${j}]}/${Models[${j}]}
#    for i in ${!file_name[@]}
#    do
#        python find_fragment.py \
#        --data_type voc \
#        --save_dir ./detect_result/find_fragment/vgg16_sigmoid_thresh_multilabel/${trainPath[${j}]} \
#        --image_dir ${file_name[${i}]} \
#        --IoU_threshold ${IoU_thresh} \
#        --confidence_threshold ${conf_thresh} \
#        --net_type ${net} \
#        --model_weight ${model} \
#        --anchors 1,1,1 \
#        --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#        --num_anchors 4,6,6,6,4,4 \
#        --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
#    done
#done
#

# # sample_find_fragment (experiment in new method to get fragment)
# train=vgg16_base
# net=vgg16-ssd
# model=./models/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# python find_fragment.py \
# --data_type voc \
# --track_Id 15 \
# --save_dir ./detect_result/find_fragment/${train} \
# --image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/bmx-bumps/ \
# --IoU_threshold ${IoU_thresh} \
# --confidence_threshold ${conf_thresh} \
# --net_type ${net} \
# --model_weight ${model} \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \



# # resnet50 - basic model
# train=resnet50_std/multi-step
# net=resnet50-ssd
# model=./models/${train}/resnet50-ssd-Epoch-199-Loss-2.8794355961584275.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --save_dir ./detect_result/find_fragment/${train}_multilabel_thresh02 \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
# done

# # resnet50 - sigmoid
# train=resnet50_std_sigmoid/sw/0406
# net=resnet50-ssd
# model=./models/${train}/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth
# for i in ${!file_name[@]}
# do
#     python find_fragment.py \
#     --data_type voc \
#     --save_dir ./detect_result/find_fragment/${train}/multilabel_thresh02 \
#     --image_dir ${file_name[${i}]} \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
# done


# train=complement_with_fractional/vgg16_basic
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.923230706491778.pth
# python find_fragment.py \
# --data_type voc \
# --save_dir ./detect_result/find_fragment/${train} \
# --image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/mbike-trick/ \
# --IoU_threshold ${IoU_thresh} \
# --confidence_threshold ${conf_thresh} \
# --net_type ${net} \
# --model_weight ${model} \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512