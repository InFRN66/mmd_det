#!/bin/bash

# for seed in 100 200 300 400 500
# do
#     python random_sample.py --seed ${seed} --path ./stat_csv/statics_anchors${seed}.csv
# done

# num_imgs=1000
# seed=1234
# for iou_thresh in 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do 
#     python random_sample.py --seed ${seed} --path ./stat_csv/statics_anchors${seed}_thresh${iou_thresh}.csv \
#     --iou_thresh ${iou_thresh} --num_imgs ${num_imgs}
# done

num_imgs=1000
seed=1234
for iou_thresh in 0.45 0.46 0.47 0.48 0.49 0.51 0.52 0.53 0.54 0.55
do 
    python random_sample.py --seed ${seed} --path ./stat_csv/statics_anchors${seed}_thresh${iou_thresh}.csv \
    --iou_thresh ${iou_thresh} --num_imgs ${num_imgs}
done
