#!/bin/bash

# saving basic-detection result
conf_thresh=0.5
IoU_thresh=0.5

declare -a file_name=(`ls -d /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/*/`)

# # --- for vgg16-ssd basic matcching IoU = 0.5
# train=vgg16_basic
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# for i in ${!file_name[@]}
# do
#     echo ${file_name[${i}]}
#     python do_detect.py \
#     --confidence_threshold ${conf_thresh} \
#     --IoU_threshold ${IoU_thresh} \
#     --image_dir ${file_name[${i}]} \
#     --save_dir ./detect_result/do_detect/aaa \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 
# done


# # --- for vgg16-ssd basic matcching IoU = 0.4
# train=compare/vgg16_IoU_04
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.989526524082307.pth
# for i in ${!file_name[@]}
# do
#     echo ${file_name[${i}]}
#     python do_detect.py \
#     --confidence_threshold ${conf_thresh} \
#     --IoU_threshold ${IoU_thresh} \
#     --image_dir ${file_name[${i}]} \
#     --save_dir ./detect_result/do_detect/${train} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 
# done



# # --- for vgg16-ssd basic complement with fractional pooling
# train=complement_with_fractional/vgg16_basic
# net=vgg16-ssd
# model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.923230706491778.pth
# for i in ${!file_name[@]}
# do
#     echo ${file_name[${i}]}
#     python do_detect.py \
#     --confidence_threshold ${conf_thresh} \
#     --IoU_threshold ${IoU_thresh} \
#     --image_dir ${file_name[${i}]} \
#     --save_dir ./detect_result/do_detect/${train} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 
# done

# trainPath=(
#     sw/0305
#     sw/035055
#     sw/03506
#     sw/035065
#     sw/04055
#     sw/045055
# )

# Models=(
#     vgg16-ssd-Epoch-200-Loss-2.7386556971457696.pth
#     vgg16-ssd-Epoch-200-Loss-2.801233918436112.pth
#     vgg16-ssd-Epoch-200-Loss-2.759871079075721.pth
#     vgg16-ssd-Epoch-200-Loss-2.7583961202252296.pth
#     vgg16-ssd-Epoch-200-Loss-2.7950822553327006.pth
#     vgg16-ssd-Epoch-200-Loss-2.6780950776992305.pth    
# )
# net=vgg16-ssd
# for i in ${!trainPath[@]}
# do
#     train=${trainPath[${i}]}
#     model=./models/vgg16_sigmoid_thresh/${train}/${Models[${i}]}
#     for i in ${!file_name[@]}
#     do
#         echo ${file_name[${i}]}
#         python do_detect.py \
#         --confidence_threshold ${conf_thresh} \
#         --IoU_threshold ${IoU_thresh} \
#         --image_dir ${file_name[${i}]} \
#         --save_dir ./detect_result/do_detect/vgg16_sigmoid_thresh/${train} \
#         --net_type ${net} \
#         --model_weight ${model} \
#         --anchors 1,1,1 \
#         --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#         --num_anchors 4,6,6,6,4,4 \
#         --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 
#     done
# done


# # --- for vgg16-ssd sigmoid thresh
# train=ww_0406/run04
# net=vgg16-ssd
# model=./models/vgg16_sigmoid_thresh/${train}/vgg16-ssd-Epoch-200-Loss-0.9991600936458956.pth
# for i in ${!file_name[@]}
# do
#     echo ${file_name[${i}]}
#     python do_detect.py \
#     --confidence_threshold ${conf_thresh} \
#     --IoU_threshold ${IoU_thresh} \
#     --image_dir ${file_name[${i}]} \
#     --save_dir ./detect_result/do_detect/vgg16_sigmoid_thresh/${train} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 
# done



# # thesis images
# conf_thresh=0.75
# # train=vgg16_basic
# train=vgg16_sigmoid_thresh/sw/0406
# net=vgg16-ssd
# # model=./models/${train}/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# model=./models/${train}/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth
# python do_detect.py \
# --confidence_threshold ${conf_thresh} \
# --IoU_threshold ${IoU_thresh} \
# --image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/bmx-bumps/ \
# --save_dir ./detect_result/do_detect/thesis_image/${train} \
# --net_type ${net} \
# --model_weight ${model} \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512


# # --- for resnet50-ssd basic 
# train=resnet50_std_sigmoid/sw/0406
# net=resnet50-ssd
# model=./models/${train}/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth
# for i in ${!file_name[@]}
# do
#     echo ${file_name[${i}]}
#     python do_detect.py \
#     --confidence_threshold ${conf_thresh} \
#     --IoU_threshold ${IoU_thresh} \
#     --image_dir ${file_name[${i}]} \
#     --save_dir ./detect_result/do_detect/${train} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 
# done


conf_thresh=0.6
# thesis image single detection
net=vgg16-ssd
# model=./models/vgg16_basic/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
model=./models/vgg16_sigmoid_thresh/sw/0406/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth

net=resnet50-ssd
model=./models/resnet50_std_sigmoid/sw/0406/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth

python thesis_image.py \
--confidence_threshold ${conf_thresh} \
--IoU_threshold ${IoU_thresh} \
--image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/soapbox/ \
--save_dir ./detect_result/do_detect/thesis_image/no_score/resnet50/sigmoid/ \
--net_type ${net} \
--model_weight ${model} \
--anchors 1,1,1 \
--rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
--num_anchors 4,6,6,6,4,4 \
--vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 


# # vgg16 - basic model

# # for i in dog longboard dog-agility libby bus swing 
# for i in dogs-scale
# do
#     net=vgg16-ssd
#     model=./models/vgg16_basic/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
#     python thesis_external.py \
#     --data_type voc \
#     --save_dir ./detect_result/find_fragment/thesis_image \
#     --image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/${i}/ \
#     --IoU_threshold ${IoU_thresh} \
#     --confidence_threshold ${conf_thresh} \
#     --net_type ${net} \
#     --model_weight ${model} \
#     --anchors 1,1,1 \
#     --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
#     --num_anchors 4,6,6,6,4,4 \
#     --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512
# done





# --------------------------------------------------------------------------------------------------------
# # for generate images

# conf_thresh=0.6

# net=vgg16-ssd
# # model=./models/vgg16_basic/vgg16-ssd-Epoch-199-Loss-2.8085033478275423.pth
# model=./models/vgg16_sigmoid_thresh/sw/0406/vgg16-ssd-Epoch-200-Loss-2.7839602047397243.pth

# net=resnet50-ssd
# model=./models/resnet50_std_sigmoid/sw/0406/resnet50-ssd-Epoch-200-Loss-2.859823820667882.pth

# python thesis_external.py \
# --data_type voc \
# --save_dir ./detect_result/find_fragment/thesis_image/resnet50/sigmoid \
# --image_dir /mnt/hdd01/img_data/DAVIS/2017/DAVIS/JPEGImages/again/bus/ \
# --id 1 \
# --IoU_threshold ${IoU_thresh} \
# --confidence_threshold ${conf_thresh} \
# --net_type ${net} \
# --model_weight ${model} \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512