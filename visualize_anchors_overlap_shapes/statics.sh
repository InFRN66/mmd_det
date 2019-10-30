#!/bin/bash

# declare -a files=(`ls ./stat_csv/`)
sep=100
seed=1234

# for iou_thresh in 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do
#     python statics.py --file stat_csv/statics_anchors${seed}_thresh${iou_thresh}.csv --sep ${sep}
# done

# for iou_thresh in 0.45 0.46 0.47 0.48 0.49 0.51 0.52 0.53 0.54 0.55
# do
#     python statics.py --file stat_csv/statics_anchors${seed}_thresh${iou_thresh}.csv --sep ${sep}
# done


python statics.py --file stat_csv/statics_anchors${seed}_thresh${iou_thresh}.csv --sep ${sep} \
--thresholds 0.4,0.5,0.6,0.7


# python statics.py --file stat_csv/statics_anchors${seed}_thresh${iou_thresh}.csv --sep ${sep} \
# --thresholds 0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55