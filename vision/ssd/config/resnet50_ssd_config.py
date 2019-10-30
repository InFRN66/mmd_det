import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


# image_size = 300

# image_mean = np.array([123, 117, 104]) # RGB layout
# # image_mean = np.array([104, 117, 123]) # BGR layout 

# image_std = 1.0

# # iou_threshold = 0.45
# iou_threshold = 0.5
# center_variance = 0.1
# size_variance = 0.2

# # anchors 466644
# specs = [
#     SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# ]

# # # for anchor 666644
# # specs = [
# #     SSDSpec(38, 8, SSDBoxSizes(30, 60), [2, 3]),
# #     SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
# #     SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
# #     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
# #     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
# #     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# # ]

# priors = generate_ssd_priors(specs, image_size)

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
        """
        print(anchors, num_anchors, rectangles, vgg_config)
        self.image_size = 300
        self.image_mean = np.array([0.485, 0.456, 0.406])*255  # RGB layout
        self.image_std = np.array([0.229, 0.224, 0.225])*255
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
        # self.specs = [
        #             SSDSpec(38, 8, SSDBoxSizes(30, 45), rectangles[0]),
        #             SSDSpec(27, 12, SSDBoxSizes(45, 60), rectangles[1]),
        #             SSDSpec(19, 16, SSDBoxSizes(60, 86), rectangles[2]),
        #             SSDSpec(14, 22, SSDBoxSizes(86, 111), rectangles[3]), 
        #             SSDSpec(10, 32, SSDBoxSizes(111, 162), rectangles[4]),
        #             SSDSpec(5, 64, SSDBoxSizes(162, 213), rectangles[5]),
        #             SSDSpec(3, 100, SSDBoxSizes(213, 264), rectangles[6]),
        #             SSDSpec(1, 300, SSDBoxSizes(264, 315), rectangles[7])
        #             ]
        self.priors = generate_ssd_priors(self.specs, self.image_size, anchors)
        self.num_anchors = num_anchors
        self.pretrained = pretrained
