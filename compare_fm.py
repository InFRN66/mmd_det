import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from math import floor, ceil

layer_keys = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
layer_channel = [512, 1024, 512, 256, 256, 256]
grid_size = [38, 19, 10, 5, 3, 1]
anchor = [4, 6, 6, 6, 4, 4]
thresh_8732 = [5776, 7942, 8542, 8692, 8728, 8732]


def row2feature(row_conf, thresh_8732=thresh_8732, grid_size=grid_size):
    '''
    row_conf : [1, 8732, 21], before softmax, all bb
        return : list of [1, anchor*21, grid, grid], shape is featuremap
    '''
    print(thresh_8732, grid_size)

    batch = row_conf.shape[0]
    # thresh_8732 = thresh_8732.insert(0, 0)
    
    featuremap = list()
    for b in range(batch):
        for i in range(len(thresh_8732)):
            if i == 0:
                x = row_conf[b, :thresh_8732[i], :]
            else:
                x = row_conf[b, thresh_8732[i-1]:thresh_8732[i], :]
            # print(b)
            x = x.view(b+1, -1)
            x = x.view(b+1, grid_size[i], grid_size[i], -1)
            x = x.permute(0,3,1,2) # [batch, ?*21, 38, 38]
            featuremap.append(x)
    return featuremap        


def mean_L1norm_per_fm(inputs1, inputs2):
    '''
    return mean L1norm between inputs1_fm and inputs2_fm per feature map.
        [512, 1024, 512, 256, 256, 256], list of tensor

    inputs1, inputs2 : list of fm, [layer][b, c, h, w]
    '''
    grid_size = [i.shape[-1]*i.shape[-2] for i in inputs1]
    # print("grid", grid_size)

    each_fm = dict()
    for l in range(len(inputs1)):
        out = [ torch.norm(inputs1[l][0,fm]-inputs2[l][0,fm], p=1)/grid_size[l] for fm in range(len(inputs1[l][0])) ]
        out = torch.FloatTensor(out)
        each_fm[l] = out
    
    each_layer = dict()
    for l in range(len(inputs1)):
        each_layer[l] = each_fm[l].mean()
    
    return each_fm, each_layer


def mean_L2norm_per_fm(inputs1, inputs2):
    '''
    return mean L2norm between inputs1_fm and inputs2_fm per feature map.
        [512, 1024, 512, 256, 256, 256], lisst of tensor

    inputs1, inputs2 : list of fm, [1, 512, 38,38], [1, 1024, 19,19], [1, 512, 10,10], [1, 256, 5,5], [1, 256, 3,3], [1, 256, 1,1]
    '''
    grid_size = [i.shape[-1]*i.shape[-2] for i in inputs1]

    each_fm = dict()
    for l in range(len(inputs1)):
        out = [ torch.norm(inputs1[l][0,fm]-inputs2[l][0,fm], p=2)/grid_size[l] for fm in range(len(inputs1[l][0]))]
        out = torch.FloatTensor(out)
        each_fm[l] = out
    
    each_layer = dict()
    for l in range(len(inputs1)):
        each_layer[l] = each_fm[l].mean()
    
    return each_fm, each_layer


class TraceBB:
    '''
    useage : 
        t = TraceBB()
        o, groups  = t.trace_BB_index(selected_index[[frame]], row_conf[[frame]], cl=15)
    '''

    def __init__(self):
        self.num_classes = 21
        self.anchor = [4, 6, 6, 6, 4, 4]
        self.grid_size = [38, 19, 10, 5, 3, 1]
        self.thresh_8732 = [5776, 7942, 8542, 8692, 8728, 8732]

    def discriminate_BBindex(self, selected_index, row_conf, cl=15):
        '''
        selected_index : [frame,21,200] // 
            in each class, index of BB in 8732 (corresponds with row_conf).
        
        row_conf : [frame, 8732, 21] //
            without softmax, confidence score in each class, BB.
        
        cl : Int // class id.

        bbN : Int // index in output(200BB) of BB to trace.
        
        return : tensor of the group to which each BB_index belongs.
            ex) selected_index : tensor([ 8619.,  8199.,  8247.,  3015.,  3003.,  ,,,
                return : tensor([ 3,  2,  2,  0,  0,  0,  0,  0,  1,  0,  0,
        '''
        mask = [selected_index[:,cl]>0][0]
        index = torch.masked_select(selected_index[:,cl], mask).type(torch.LongTensor)
        selected_conf = row_conf[:, index] # ex) [53, 21]

        groups = torch.zeros_like(index)
        for i, idx in enumerate(index):
            if 0 <= idx < (38**2)*4:
                groups[i] = 0

            elif (38**2)*4 <= idx < (38**2)*4+(19**2)*6:
                groups[i] = 1
            
            elif (38**2)*4+(19**2)*6 <= idx < (38**2)*4+(19**2)*6+(10**2)*6:
                groups[i] = 2

            elif (38**2)*4+(19**2)*6+(10**2)*6 <= idx < (38**2)*4+(19**2)*6+(10**2)*6+(5**2)*6:
                groups[i] = 3
                
            elif (38**2)*4+(19**2)*6+(10**2)*6+(5**2)*6 <= idx < (38**2)*4+(19**2)*6+(10**2)*6+(5**2)*6+(3**2)*4:
                groups[i] = 4
                
            elif (38**2)*4+(19**2)*6+(10**2)*6+(5**2)*6+(3**2)*4 <= idx < (38**2)*4+(19**2)*6+(10**2)*6+(5**2)*6+(3**2)*4+(1**2)*4:
                groups[i] = 5
            else:
                raise Exception('out of index.')
        return index, groups


    def convertIndex_outer2iner(self, indicies, groups):
        '''
        convert index with 8732 to index with its group's box
            ex) if index = 8119(in 8732 order), then group is 2.
                In group 2, 8119 is 8119 - 7942 = 257th(in group 2 order).
        '''
        # print(index, groups)
        inner = [ indicies[i]-self.thresh_8732[groups[i]-1] if groups[i]>0 else indicies[i] \
                    for i in range(len(indicies)) ]
        return torch.LongTensor(inner)


    # def trace_BB_index(self, index, group, batch_size=1, anchor=anchor):
    #     '''
    #     index : Int,
    #     group : Int,
    #     return : index of a grid which outputs the BB.
    #     '''
    #     frame = np.arange(0,grid_size[group]**2*anchor[group])
    #     frame = frame.reshape((batch_size, grid_size[group], grid_size[group], anchor[group])).transpose(0,3,1,2)
    #     # print('frame_shape: ', '\n', frame.shape)
    #     # print('frame :', frame)
    #     batch, anchor, h, w = np.where(frame==index)
    #     # print(index)
    #     # print('frame==index', frame == index)
    #     # print(batch, anchor, h, w)
        
    #     return int(batch), int(anchor), int(h), int(w)

    def trace_BB_index(self, selected_index, row_conf, cl):
        '''
        index : Int,
        group : Int,
        return : index of a grid which outputs the BB.
        '''
        indicies, groups = self.discriminate_BBindex(selected_index, row_conf, cl)
        inner_indicies = self.convertIndex_outer2iner(indicies, groups)

        output = torch.zeros(len(groups), 4)
        for i, (index, group) in enumerate(zip(inner_indicies, groups)):
            frame = np.arange(0, (self.grid_size[group]**2)*self.anchor[group])
            frame = frame.reshape((1, self.grid_size[group], self.grid_size[group], self.anchor[group])).transpose(0,3,1,2)
            frame, anchor, h, w = np.where(frame==index)
            output[i, :] = torch.Tensor([int(frame), int(anchor)*self.num_classes+cl, int(h), int(w)])
            # print(index)
            # print('frame==index', frame == index)
            # print(batch, anchor, h, w)
        
        # return int(batch), int(anchor), int(h), int(w)
        return output, groups, indicies



class Rendering:
    '''
    specify the feature map area in which target BB exists.
    '''
    def __init__(self, inputs1, inputs2):
        # self.layer_keys = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
        # self.layer_channel = [512, 1024, 512, 256, 256, 256]
        # self.grid_size = [38, 19, 10, 5, 3, 1]
        self.inputs1 = inputs1
        self.inputs2 = inputs2
        self.grid_size = [ inputs1[l].shape[-1] for l in range(len(inputs1)) ]


    def target_rendering(self, output, cl, bbN):
        # sepecify human area to near bb
        target = output[0, cl][bbN]# _, x1, y1, x2, y2

        coodinates = dict()
        # 6 scale size
        coodinates['x1'] = [ floor(self.grid_size[l] * target[1].item()) for l in range(len(self.grid_size)) ]
        coodinates['y1'] = [ floor(self.grid_size[l] * target[2].item()) for l in range(len(self.grid_size)) ]
        coodinates['x2'] = [ ceil(self.grid_size[l] * target[3].item()) for l in range(len(self.grid_size)) ]
        coodinates['y2'] = [ ceil(self.grid_size[l] * target[4].item()) for l in range(len(self.grid_size)) ]
        # print('xy :', scale_x1, '\n', scale_x2, '\n', scale_y1, '\n', scale_y2)
        return coodinates


    def rendering(self, output1, output2, cl=15, bbN=None):
        if type(bbN) is not list:
            raise Exception("bbN must be a list.")

        out_coord = dict()
        coord1 = self.target_rendering(output1, cl, bbN[0])
        coord2 = self.target_rendering(output2, cl, bbN[1])

        for key in coord1:
            array = np.vstack((coord1[key], coord2[key]))
            # print(array.shape)
            if key in ['x1', 'y1']:
                array = np.min(array, axis=0)
            elif key in ['x2', 'y2']:
                array = np.max(array, axis=0)
            # print(array.shape)
            out_coord[key] = array

        rend_fm1 = [ self.inputs1[l][:, :, out_coord['y1'][l]:out_coord['y2'][l], out_coord['x1'][l]:out_coord['x2'][l] ] \
                                for l in range(len(self.inputs1))]

        rend_fm2 = [ self.inputs2[l][:, :, out_coord['y1'][l]:out_coord['y2'][l], out_coord['x1'][l]:out_coord['x2'][l] ] \
                                for l in range(len(self.inputs1))]

        return rend_fm1, rend_fm2





##################################






class Rendering2:
    '''
    specify the feature map area in which target BB exists.
    '''
    def __init__(self, inputs1, inputs2):
        # self.layer_keys = ['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
        # self.layer_channel = [512, 1024, 512, 256, 256, 256]
        # self.grid_size = [38, 19, 10, 5, 3, 1]
        self.inputs1 = inputs1
        self.inputs2 = inputs2
        self.grid_size = [ inputs1[l].shape[-1] for l in range(len(inputs1)) ]


    def target_rendering(self, output, cl, bbN):
        # sepecify human area to near bb
        target = output[0, cl][bbN]# _, x1, y1, x2, y2

        coodinates = dict()
        # 6 scale size
        coodinates['x1'] = [ floor(self.grid_size[l] * target[1].item()) for l in range(len(self.grid_size)) ]
        coodinates['y1'] = [ floor(self.grid_size[l] * target[2].item()) for l in range(len(self.grid_size)) ]
        coodinates['x2'] = [ ceil(self.grid_size[l] * target[3].item()) for l in range(len(self.grid_size)) ]
        coodinates['y2'] = [ ceil(self.grid_size[l] * target[4].item()) for l in range(len(self.grid_size)) ]
        # print('xy :', scale_x1, '\n', scale_x2, '\n', scale_y1, '\n', scale_y2)
        return coodinates


    def rendering(self, output1, output2, cl=15, bbN=None):
        if type(bbN) is not list:
            raise Exception("bbN must be a list.")

        out_coord = dict()
        coord1 = self.target_rendering(output1, cl, bbN[0])
        coord2 = self.target_rendering(output2, cl, bbN[1])

        for key in coord1:
            array = np.vstack((coord1[key], coord2[key]))
            # print(array.shape)
            if key in ['x1', 'y1']:
                array = np.min(array, axis=0)
            elif key in ['x2', 'y2']:
                array = np.max(array, axis=0)
            # print(array.shape)
            out_coord[key] = array

        rend_fm1 = [ self.inputs1[l][:, :, out_coord['y1'][l]:out_coord['y2'][l], out_coord['x1'][l]:out_coord['x2'][l] ] \
                                for l in range(len(self.inputs1))]

        rend_fm2 = [ self.inputs2[l][:, :, out_coord['y1'][l]:out_coord['y2'][l], out_coord['x1'][l]:out_coord['x2'][l] ] \
                                for l in range(len(self.inputs1))]

        return rend_fm1, rend_fm2

