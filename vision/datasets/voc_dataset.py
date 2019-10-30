import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os, sys, json, pickle, glob

class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, return_id=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.return_id = return_id
        self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            if self.target_transform.sigmoid_thresh:
                boxes, labels, mask = self.target_transform(boxes, labels)
                return image, boxes, labels, mask
            else:
                boxes, labels = self.target_transform(boxes, labels)
    
        if self.return_id:
            return image, boxes, labels, image_id

        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            bbox = object.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



# my own original dataset class. (In case of use other data except to VOC)
class ORG_Dataset():
    def __init__(self, root, annot_root, transform=None, RGB=True, data_type='DAVIS', multiple=False):
        '''
        Args : 
            root : path to images dir 
            annot_dir : path to annotation dir
            transforms : TODO
            RGB : need or not to convert RGB from BGR.
            data_type : data type to use.
        '''
        self.root = root
        self.annot_root = annot_root
        self.data_type = data_type
        self.RGB = RGB
        self.transform = transform
        self._imglists()
        self.class_names = self.class_names = ('BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat',
                                                'bottle', 'bus', 'car', 'cat', 'chair',
                                                'cow', 'diningtable', 'dog', 'horse',
                                                'motorbike', 'person', 'pottedplant',
                                                'sheep', 'sofa', 'train', 'tvmonitor'
                                                )
        self.multiple = multiple

    def __getitem__(self, index):
        imgs = self._get_images(index)
        if self.multiple:
            coords, labels = self._get_annotation_multiple(index)
        else:
            coords, labels = self._get_annotation(index)
        return imgs, coords, labels # [h, w, c], [num_bb, 4], [num_bb]
        
    
    def _get_images(self, index):
        img = cv2.imread(self.imglists[index])
        if self.RGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    # --- old, one class from label.json
    def _get_annotation(self, index):
        if self.data_type == 'DAVIS':
            annotation = self._davis_annotation()
            frame_info = annotation[str(index)]
            coords = np.array([[frame_info['x1'], frame_info['y1'], frame_info['x2'], frame_info['y2']]]).astype(np.float)
            labels = np.array([frame_info['label']])
            return coords, labels

    # --- new label, multiple classes
    def _get_annotation_multiple(self, index):
        if self.data_type == 'DAVIS':
            annotation = self._davis_annotation_multiple()
            frame_info = annotation[index]
            coords = frame_info["box"]
            labels = frame_info["label"]
            return coords, labels # np.array(num_object, 4), np.array(num_object)

    # --- 
    def _davis_annotation(self):
        with open(self.annot_root, 'r') as f:
            json_file = json.load(f)
            return json_file

    def _davis_annotation_multiple(self):
        print("f: {}".format(self.annot_root))
        with open(self.annot_root, 'rb') as f:
            pkl_file = pickle.load(f)
            return pkl_file
        
        
    def _imglists(self):
        imglists = glob.glob(self.root+'*.jpg') # TODO remove not_Img file
        imglists.sort()
        self.imglists = imglists

    
    def __len__(self):
        return len(self.imglists)




# class VOCDataset_Scale:

#     def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False):
#         """Dataset for VOC data.
#         Args:
#             root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
#                 Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.

#         In my case first(datasets): root = VOdevkit/VOC2007  / is_test =False / 
#         In my case second(validation_dataset): root = VOdevkit/VOC2007  / is_test =False / 
#         """
#         self.root = pathlib.Path(root)
#         self.transform = transform
#         self.target_transform = target_transform
#         if is_test:
#             image_sets_file = self.root / "ImageSets/Main/test.txt" # test.txt (4952)
#         # train : X
#         else:
#             image_sets_file = self.root / "ImageSets/Main/trainval.txt" # trainval.txt (5011)
#         self.ids = VOCDataset._read_image_ids(image_sets_file) # load all ids in trainval.txt
#         self.keep_difficult = keep_difficult

#         self.class_names = ('BACKGROUND',
#             'aeroplane', 'bicycle', 'bird', 'boat',
#             'bottle', 'bus', 'car', 'cat', 'chair',
#             'cow', 'diningtable', 'dog', 'horse',
#             'motorbike', 'person', 'pottedplant',
#             'sheep', 'sofa', 'train', 'tvmonitor'
#         )
#         self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

#     def __getitem__(self, index):
#         image_id = self.ids[index]
#         boxes, labels, is_difficult = self._get_annotation(image_id)
#         if not self.keep_difficult:
#             boxes = boxes[is_difficult == 0]
#             labels = labels[is_difficult == 0]
#         image = self._read_image(image_id)
        
#         # hw_scale -> self.transform(TrainAugmentation in data_preprocessing) -> normalized_scale
#         if self.transform:
#             image, box, label, IOU, ret = self.transform(image, boxes, labels) # :: change here
    
#         # convert to MatchPriors in ssd.ssd.MatchPrior
#         if self.target_transform:
#             # print('in target transform ')
#             # # print('img', image.shape, image) # [3, 300, 300]
#             # print('box', box.shape, box) # [num, 4]
#             # print('label', label.shape, label) # [num]
#             # print()
#             box, label = self.target_transform(box, label)

#         # return image, boxes, labels, scale_coeff, anchors, IOU
#         return image, box, label, IOU, ret

#     def get_image(self, index):
#         image_id = self.ids[index]
#         image = self._read_image(image_id)
#         if self.transform:
#             image, _ = self.transform(image)
#         return image

#     def get_annotation(self, index):
#         image_id = self.ids[index]
#         return image_id, self._get_annotation(image_id)

#     def __len__(self):
#         return len(self.ids)

#     @staticmethod
#     def _read_image_ids(image_sets_file):
#         ids = []
#         with open(image_sets_file) as f:
#             for line in f:
#                 ids.append(line.rstrip())
#         return ids

#     def _get_annotation(self, image_id):
#         annotation_file = self.root / f"Annotations/{image_id}.xml"
#         objects = ET.parse(annotation_file).findall("object")
#         boxes = []
#         labels = []
#         is_difficult = []
#         for object in objects:
#             class_name = object.find('name').text.lower().strip()
#             bbox = object.find('bndbox')
#             # VOC dataset format follows Matlab, in which indexes start from 0
#             x1 = float(bbox.find('xmin').text) - 1
#             y1 = float(bbox.find('ymin').text) - 1
#             x2 = float(bbox.find('xmax').text) - 1
#             y2 = float(bbox.find('ymax').text) - 1
#             boxes.append([x1, y1, x2, y2])
#             labels.append(self.class_dict[class_name])
#             is_difficult_str = object.find('difficult').text
#             is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

#         return (np.array(boxes, dtype=np.float32),
#                 np.array(labels, dtype=np.int64),
#                 np.array(is_difficult, dtype=np.uint8))

#     def _read_image(self, image_id):
#         image_file = self.root / f"JPEGImages/{image_id}.jpg"
#         image = cv2.imread(str(image_file))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image



# # default augment, scale augment 混合
# class VOCDataset_plusScale:

#     def __init__(self, root, default_transform=None, scale_transform=None, target_transform=None, is_test=False, keep_difficult=False):
#         """Dataset for VOC data.
#         Args:
#             root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
#                 Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.

#         In my case first(datasets): root = VOdevkit/VOC2007  / is_test =False / 
#         In my case second(validation_dataset): root = VOdevkit/VOC2007  / is_test =False / 
#         """
#         self.root = pathlib.Path(root)
#         self.default_transform = default_transform
#         self.scale_transform = scale_transform
#         self.target_transform = target_transform
#         if is_test:
#             image_sets_file = self.root / "ImageSets/Main/test.txt" # test.txt (4952)
#         # train : X
#         else:
#             image_sets_file = self.root / "ImageSets/Main/trainval.txt" # trainval.txt (5011)
#         self.ids = VOCDataset._read_image_ids(image_sets_file) # load all ids in trainval.txt
#         self.keep_difficult = keep_difficult

#         self.class_names = ('BACKGROUND',
#             'aeroplane', 'bicycle', 'bird', 'boat',
#             'bottle', 'bus', 'car', 'cat', 'chair',
#             'cow', 'diningtable', 'dog', 'horse',
#             'motorbike', 'person', 'pottedplant',
#             'sheep', 'sofa', 'train', 'tvmonitor'
#         )
#         self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

#     def __getitem__(self, index):
#         image_id = self.ids[index]
#         boxes, labels, is_difficult = self._get_annotation(image_id)
#         if not self.keep_difficult:
#             boxes = boxes[is_difficult == 0]
#             labels = labels[is_difficult == 0]
#         image = self._read_image(image_id)
#         # print('sksksksksks', image.shape, boxes.shape, labels.shape)
#         # hw_scale -> self.transform(TrainAugmentation in data_preprocessing) -> normalized_scale
#         trans_random = np.random.randint(2) # 0 or 1
#         trans_random = 1
#         # print('transform set : ', trans_random)
#         if trans_random == 0 and self.default_transform:
#             images, boxes, labels = self.default_transform(image, boxes, labels) # :: TrainAugmentation
        
#         elif trans_random == 1 and self.scale_transform:
#             images, boxes, labels, IOUs, rets = self.scale_transform(image, boxes, labels) # :: Train_ScaleAugmentation_for_Retrain

#         elif self.default_transform == None and self.scale_transform:
#             images, boxes, labels, IOUs, rets = self.scale_transform(image, boxes, labels) # :: Train_ScaleAugmentation_for_Retrain only
    
#         # print('$$$$$', image.shape, boxes.shape, labels.shape)
#         # convert to MatchPriors in ssd.ssd.MatchPrior
#         if self.target_transform:
#             boxes, labels = self.target_transform(boxes, labels)

#         # return image, boxes, labels, scale_coeff, anchors, IOU
#         return images, boxes, labels

#     def get_image(self, index):
#         image_id = self.ids[index]
#         image = self._read_image(image_id)
#         if self.default_transform:
#             image, _ = self.default_transform(image)
#         return image

#     def get_annotation(self, index):
#         image_id = self.ids[index]
#         return image_id, self._get_annotation(image_id)

#     def __len__(self):
#         return len(self.ids)

#     @staticmethod
#     def _read_image_ids(image_sets_file):
#         ids = []
#         with open(image_sets_file) as f:
#             for line in f:
#                 ids.append(line.rstrip())
#         return ids

#     def _get_annotation(self, image_id):
#         annotation_file = self.root / f"Annotations/{image_id}.xml"
#         objects = ET.parse(annotation_file).findall("object")
#         boxes = []
#         labels = []
#         is_difficult = []
#         for object in objects:
#             class_name = object.find('name').text.lower().strip()
#             bbox = object.find('bndbox')
#             # VOC dataset format follows Matlab, in which indexes start from 0
#             x1 = float(bbox.find('xmin').text) - 1
#             y1 = float(bbox.find('ymin').text) - 1
#             x2 = float(bbox.find('xmax').text) - 1
#             y2 = float(bbox.find('ymax').text) - 1
#             boxes.append([x1, y1, x2, y2])
#             labels.append(self.class_dict[class_name])
#             is_difficult_str = object.find('difficult').text
#             is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

#         return (np.array(boxes, dtype=np.float32),
#                 np.array(labels, dtype=np.int64),
#                 np.array(is_difficult, dtype=np.uint8))

#     def _read_image(self, image_id):
#         image_file = self.root / f"JPEGImages/{image_id}.jpg"
#         image = cv2.imread(str(image_file))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image


class DAVISDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, return_id=False):
        """Dataset for DAVIS data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.return_id = return_id
        self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        # print('ids', self.ids)
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
    
        if self.return_id:
            return image, boxes, labels, image_id

        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            bbox = object.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
