import numpy as np

from vision.utils.box_utils import (SSDSpec, SSDBoxSizes,
                                     generate_ssd_priors, generate_ssd_priors_custom)


# image_size = 300
# image_mean = np.array([123, 117, 104])  # RGB layout
# image_std = 1.0

# # iou_threshold = 0.45
# iou_threshold = 0.5
# center_variance = 0.1
# size_variance = 0.2

# specs = [
#     SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# ]

# # for anchor 023 or 123
# specs = [
#     SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(19, 16, SSDBoxSizes(60, 111), [2]),
#     SSDSpec(10, 32, SSDBoxSizes(111, 162), [2]),
#     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2]),
#     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# ]

# priors = generate_ssd_priors(specs, image_size, anchors=[1,1,1]) # small, big, rects
# priors = generate_ssd_priors_custom(specs, image_size)


# 'SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios']
# --- object style config ---
class Config:
    def __init__(self, anchors, rectangles, num_anchors, vgg_config, pretrained=None):
        """

        [from vision.ssd.config import vgg_ssd_config]

        to 
        
        [from vision.ssd.config.vgg_ssd_config import Config]
        [config = Config(anchors, rectangles, num_anchors)]
        
        Args:: 
            default:
                anchors     = [1,1,1] / small_square / big_square / rects
                num_anchors = [4,6,6,6,4,4]
                rectangles  = [[2],[2,3],[2,3],[2,3],[2],[2]]
                vgg_config  = [64, 64, 'M', 
                               128, 128, 'M',
                               256, 256, 256, 'C',
                               512, 512, 512, 'M', 
                               512, 512, 512]      conv1 / conv2 / conv3 / conv4 /conv5
                pretrained  = ./models/vgg16_reducedfc.pth  or None / need only if want to use fractional Pooling, for train
        """
        self.image_size = 300
        self.image_mean = np.array([123, 117, 104])  # RGB layout
        self.image_std = 1.0
        self.iou_threshold = 0.5
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.vgg_config = vgg_config
        self.specs = [
                    SSDSpec(38, 8, SSDBoxSizes(30, 60), rectangles[0]),
                    SSDSpec(19, 16, SSDBoxSizes(60, 111), rectangles[1]),
                    SSDSpec(10, 32, SSDBoxSizes(111, 162), rectangles[2]),
                    SSDSpec(5, 64, SSDBoxSizes(162, 213), rectangles[3]),
                    SSDSpec(3, 100, SSDBoxSizes(213, 264), rectangles[4]),
                    SSDSpec(1, 300, SSDBoxSizes(264, 315), rectangles[5])
                    ]

        self.priors = generate_ssd_priors(self.specs, self.image_size, anchors)
        self.num_anchors = num_anchors
        self.pretrained = pretrained
# --- *** --- *** ---

class Config_complement_with_fractional(Config):
    def __init__(self, anchors, rectangles, num_anchors, vgg_config, pretrained=None):
        super(Config_complement_with_fractional, self).__init__(anchors, rectangles, num_anchors, vgg_config, pretrained)
        self.specs = [
            SSDSpec(38, 8, SSDBoxSizes(30, 60), rectangles[0]),
            SSDSpec(28, 11, SSDBoxSizes(45, 85), rectangles[0]),
            SSDSpec(19, 16, SSDBoxSizes(60, 111), rectangles[1]),
            SSDSpec(15, 20, SSDBoxSizes(85, 135), rectangles[1]),
            SSDSpec(10, 32, SSDBoxSizes(111, 162), rectangles[2]),
            SSDSpec(7, 43, SSDBoxSizes(135, 187), rectangles[2]),
            SSDSpec(5, 64, SSDBoxSizes(162, 213), rectangles[3]),
            SSDSpec(4, 75, SSDBoxSizes(187, 238), rectangles[3]),
            SSDSpec(3, 100, SSDBoxSizes(213, 264), rectangles[4]),
            SSDSpec(2, 150, SSDBoxSizes(238, 290), rectangles[4]),
            SSDSpec(1, 300, SSDBoxSizes(264, 315), rectangles[5])
            ]
        self.priors = generate_ssd_priors(self.specs, self.image_size, anchors)