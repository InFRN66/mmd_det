import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys, pickle, json, glob
import cv2
from collections import namedtuple, defaultdict
# import matplotlib.pyplot as plt
import linecache


def read_mAP(base_dir, require_IoU, matching_IoU):
    file_ = os.path.join(base_dir, str(matching_IoU), 'mAPIoU_mAP_graph', f'x{str(require_IoU)}-ymAP', 'all.txt')
    mAP = linecache.getline(file_, 24)
    print(file_)
    mAP = float(mAP.split(':')[-1])
    return mAP


def get_mAP(matching_IoUs, require_IoUs, base_dir='./eval_results/matching_IoU/vgg16-ssd/'):
    '''
    Args: 
        matching_IoUs: used IoUs to train the model.  e.g.) [0.1, 0.2, 0.3, 0.4, 0.5]
        require_IoUs: used IoUs in evaluation of mAP.  e.g.) [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    '''
    mAPs = dict()

    for r in require_IoUs:
        model = list()
        for m in matching_IoUs:
            mAP = read_mAP(base_dir=base_dir, matching_IoU=m, require_IoU=r)
            model.append(mAP)
        mAPs[r] = model

    # plt.figure(figsize=(15, 10))
    # plt.rcParams['font.size']=13
    # for r in require_IoUs:
    #     data = mAPs[r]
    #     plt.plot(np.linspace(0.1, 0.5, len(data)), np.array(data), marker='s', label=f'require_IoU={r}', alpha=0.8)
    # plt.grid()
    # plt.ylim(0.0, 0.9)
    # plt.xlabel('+matching_IoU[50epochs]')
    # plt.ylabel('mAP')
    # # plt.hlines(y=mAPs_base[0.5], xmin=0.1, xmax=0.9, linestyles='dashed', alpha=0.3, label=f'')
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    # plt.subplots_adjust(left=0.1, right=0.8)
    # plt.title('reinforce-diff-matchingIoU')

    # print(mAPs[0.5])
    # reinforce_05 = mAPs[0.5]
    # all_05['reinforce-diff-matchingIoU'] = mAPs[0.5]
    # plt.savefig('./mAP_IoU_result/reinforce-diff-matchingIoU-[+mIoU-mAP].png')


def show_train_val_plot(file_path, target='train'):
    """
    Args:
        flie_path: file path to {target}_losses.pickle
        target: train, or val 
    Usage:
        `show_train_val_plot(~~losses.pickle, target='train')`
    """
    if target == 'train':
        with open(file_path, 'rb') as f:
            train_plot = pickle.load(f)
        Loss, RLoss, CLoss = list(), list(), list()
        for i in range(len(train_plot)):
            data = train_plot[i]
            Loss.append((data[0], data[1]['AverageLoss']))
            RLoss.append((data[0], data[1]['AverageRegressionLoss']))
            CLoss.append((data[0], data[1]['AverageClassificationLoss']))
        Loss = np.array(Loss)
        RLoss = np.array(RLoss)
        CLoss = np.array(CLoss)

        plt.plot(Loss[:,0], Loss[:, 1], label=f'{target}_average')
        plt.plot(RLoss[:,0], RLoss[:, 1], label=f'{target}_regression')
        plt.plot(CLoss[:,0], CLoss[:, 1], label=f'{target}_classification')
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')


        # plt.savefig('loss.png')

    elif target == 'val':
        with open(file_path, 'rb') as f:
            train_plot = pickle.load(f)
        Loss, RLoss, CLoss = list(), list(), list()
        for i in range(len(train_plot)):
            data = train_plot[i]
            Loss.append((data[0], data[1]['ValLoss']))
            RLoss.append((data[0], data[1]['ValRegressionLoss']))
            CLoss.append((data[0], data[1]['ValClassificationLoss']))
        
        Loss = np.array(Loss)
        RLoss = np.array(RLoss)
        CLoss = np.array(CLoss)

        plt.plot(Loss[:,0], Loss[:, 1], label=f'{target}_average')
        plt.plot(RLoss[:,0], RLoss[:, 1], label=f'{target}_regression')
        plt.plot(CLoss[:,0], CLoss[:, 1], label=f'{target}_classification')
        plt.grid()
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')

        # plt.savefig('loss.png')
    
    else:
        raise Exception('invalid target type')


def write_grid(img, feature_grid):
    """
    Args: 
        img: [h,w,c]
        feature_grid: int, 出力格子の数
    imgに格子線を描画して返す
    """
    h, w, _ = img.shape
    h_length = h / feature_grid
    w_length = w / feature_grid
    for i in range(feature_grid):
        print(int(h_length*(i+1)))
        img = cv2.line(img, (0, int(h_length*(i+1))), (w, int(h_length*(i+1))), (255,255,255), 1)
        img = cv2.line(img, (int(w_length*(i+1)), 0), (int(w_length*(i+1)), h), (255,255,255), 1)
    return img


class Fragment_counter:
    def __init__(self, root, method, matching_IoU):
        """
        Args: example
            root: '/mnt/disk1/img_data/VOC_random_seed/grid_3pix/seed3'
            method: matching_IoU or reinforce- ... or sin... 
            matching_IoU: 0.1 or 0.2 or ...
        """
        self.root = root
        self.method = method
        self.matching_IoU = matching_IoU
        self.fragment_image_list = defaultdict(dict)
        self.fragment_boxes_list = defaultdict(dict)
        self.num = 0 # 総フレーム
        self.fragment = 0 # 総fragment
        self.miss = 0 # 総detect miss
        
        
    def load_json(self, ID, objectN):
        """
        return loaded json object for 'DAVIS_flicker.json'
        """
        if self.method == 'RefineDet320':
            with open(os.path.join(self.root, ID, self.method, objectN, 'DAVIS_flicker.json'), 'r') as f:
                JSON = json.load(f)    
        else:      
            with open(os.path.join(self.root, ID, self.method, self.matching_IoU, objectN, 'DAVIS_flicker.json'), 'r') as f:
                JSON = json.load(f)
        return JSON
    
    
    def get_image_list(self, ID):
        """
        Args: 
            ID: 006178_list3073 etc
        """
        # print(ID)
        if self.method == 'RefineDet320':
            objects = glob.glob(os.path.join(self.root, ID, self.method, '*'))
        else:
            objects = glob.glob(os.path.join(self.root, ID, self.method, self.matching_IoU, '*'))
        objects = [ x.split('/')[-1] for x in objects ]
        objects = sorted(objects, key=lambda x: int(x[6:]))
        for objectN in objects:
            # print(objectN)
            jsondata = self.load_json(ID, objectN)
            indicies = np.array(jsondata['flicker_indicies'])
            indicies = np.where(indicies==1)[0]
            # print(indicies)
            image_list = glob.glob(os.path.join(self.root, ID, '*.png'))
            image_list = sorted(image_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_list = np.array(image_list)
    #         flicker_image_list = image_list[indicies]
            image_lists = list()
            boxes_lists = list()
            for idx in indicies:
                # -2 - +2
                image_lists.append(image_list[idx-2:idx+3].tolist()) if (idx > 0 and idx < len(image_list)) else image_lists.append(image_list[idx:idx+2])
                if self.method != 'RefineDet320':
                    boxes_lists.append(jsondata['selected_indicies'][idx-2:idx+3])
            self.fragment_image_list[ID][objectN] = image_lists
            if self.method != 'RefineDet320':
                self.fragment_boxes_list[ID][objectN] = boxes_lists
            # print()
    
    
    def count(self, ID):
        """
        Args: 
            ID: 006178_list3073 etc
        """
        if self.method == 'RefineDet320':
            objects = glob.glob(os.path.join(self.root, ID, self.method, '*'))
        else:
            objects = glob.glob(os.path.join(self.root, ID, self.method, self.matching_IoU, '*'))
        objects = [ x.split('/')[-1] for x in objects ]
        objects = sorted(objects, key=lambda x: int(x[6:]))
        for objectN in objects:
            jsondata = self.load_json(ID, objectN)
            self.fragment += jsondata['num_flicker']
            self.miss += sum(jsondata['miss'])
            self.num += jsondata['num_frame']


    def dict_sort(self, DICT):
        """
        sort dict along object(int).
        """
        for key in DICT.keys():
            print(DICT[key])
            DICT[key] = sorted(DICT[key].items(), key=lambda x: int(x[0][6:]))
        return DICT

           
    def save_fragmentimages_aspickle(self, save_dir):
        """
        Args: 
            save_dir: ./record/pickle_data/voc-various
        """
        self.fragment_image_list = self.dict_sort(self.fragment_image_list)
        
        with open(os.path.join(save_dir, f'{self.method}_{self.matching_IoU}_voc.pickle'), 'wb') as f:
            pickle.dump(self.fragment_image_list, f)
        print('saved fragment image list.')