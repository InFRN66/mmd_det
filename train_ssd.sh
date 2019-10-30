#!/bin/bash

# for vgg16 - basic
python train_ssd.py \
--dataset_type voc \
--datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
--net vgg16-ssd \
--validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
--batch_size 32 \
--num_epochs 200 \
--validation_epochs 50 \
--scheduler multi-step \
--milestones 120,160 \
--lr 0.001 \
--t_max 200 \
--base_net ./models/vgg16_reducedfc.pth \
--anchors 1,1,1 \
--rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
--num_anchors 4,6,6,6,4,4 \
--vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
--checkpoint_folder ./models/compare/vgg16_IoU_04 \
--matching_IoU 0.4


# # --- for vgg16-ssd for sigmoid threshold (sampling)
# python train_ssd.py  >> ./models/vgg16_sigmoid_thresh/ww_0305/output.txt \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net vgg16-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 300 \
# --validation_epochs 50 \
# --scheduler multi-step \
# --milestones 120,160 \
# --lr 0.001 \
# --t_max 200 \
# --base_net ./models/vgg16_reducedfc.pth \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --alpha 0.3 \
# --beta 0.5 \
# --alpha_y 0.001 \
# --checkpoint_folder ./models/vgg16_sigmoid_thresh/ww_0305


# # --- for vgg16-ssd + fractional pooling w/ (conv + relu) 
# python train_ssd.py >> ./models/vgg16_fracP_withConvReLU/output.txt \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net vgg16-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 300 \
# --validation_epochs 50 \
# --scheduler multi-step \
# --milestones 120,160 \
# --lr 0.001 \
# --t_max 200 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,"F11",64,"F12",128,128,"F21",128,"F22",256,256,256,"F31",256,"F32",512,512,512,"F41",512,"F42",512,512,512 \
# --partialPT ./models/vgg16_reducedfc.pth \
# --checkpoint_folder ./models/vgg16_fracP_withConvReLU


# # --- for vgg16-ssd + fractional pooling w/o (conv + relu) 
# python train_ssd.py >> ./models/vgg16_fracP_woConvReLU/output.txt \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net vgg16-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 200 \
# --validation_epochs 50 \
# --scheduler multi-step \
# --milestones 120,160 \
# --lr 0.001 \
# --t_max 200 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,"F11","F12",128,128,"F21","F22",256,256,256,"F31","F32",512,512,512,"F41","F42",512,512,512 \
# --partialPT ./models/vgg16_reducedfc.pth \
# --checkpoint_folder ./models/vgg16_fracP_woConvReLU


# # --- for resnet50ssd + L2Norm(implemented as original class) + multistep120,160
# python train_ssd.py \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net resnet50-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 200 \
# --validation_epochs 25 \
# --scheduler multi-step \
# --milestones 120,160 \
# --lr 0.001 \
# --t_max 200 \
# --checkpoint_folder ./models/resnet50_L2Norm_multistep120160
# # --base_net ./models/vgg16_reducedfc.pth \


# # resnet50 sigmoid - scheduler = CosineAnearing
# python train_ssd.py >> ./models/resnet50_sigmoid/ww_0406_cosine/output.txt \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net resnet50-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 200 \
# --validation_epochs 50 \
# --scheduler cosine \
# --lr 0.001 \
# --t_max 200 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --alpha 0.4 \
# --beta 0.6 \
# --alpha_y 0.001 \
# --checkpoint_folder ./models/resnet50_sigmoid/ww_0406_cosine


# # resnet basic_ResNet_v2
# method=multi-step
# folder=resnet50_std/${method}
# python train_ssd.py >> models/${folder}/output.txt \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net resnet50-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 200 \
# --validation_epochs 50 \
# --scheduler ${method} \
# --milestones 120,160 \
# --lr 0.001 \
# --t_max 200 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --checkpoint_folder models/${folder}


# # for vgg16 - basic complement with pooling
# python train_ssd_complement_with_fractional.py \
# --dataset_type voc \
# --datasets /mnt/hdd01/img_data/VOCdevkit/VOC2007 /mnt/hdd01/img_data/VOCdevkit/VOC2012 \
# --net vgg16-ssd \
# --validation_dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --batch_size 32 \
# --num_epochs 200 \
# --validation_epochs 50 \
# --scheduler multi-step \
# --milestones 120,160 \
# --lr 0.001 \
# --t_max 200 \
# --base_net ./models/vgg16_reducedfc.pth \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --checkpoint_folder ./models/complement_with_fractional/vgg16_basic