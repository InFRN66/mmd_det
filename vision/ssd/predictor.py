import torch
import numpy as np
from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer

def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    # .numel : [num_bb, 4] => num_bb*4
    if boxes.numel() == 0: ## boxが空ならreturn
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order(小さい順) idx :  scoreが高いボックスのindex(score order)
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1: # 残り要素が1個になったらbreak?
            break
        idx = idx[:-1]  # remove kept element from view
        
        
        # load bboxes of next highest vals
        torch.index_select(x1, dim=0, index=idx, out=xx1) # dimに沿ってindexのelementのみ出す
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2) # 最大のボックス(i)を除いた，他の全ボックスのx1, y1, ,,...
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i]) # 
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou いま注目する（確率大きい順に注目）ボックスと，それ以外のボックスを比較したIoUなので[残り-1]
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    # print("aaa", keep, keep.shape, count)
    return keep, count


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.5, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        # print(f'prepare ::: {size, mean, std}')
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()
        self.timer = Timer()


    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes, _ = self.net.forward(images)
            print("Inference time: ", self.timer.end())

        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold # 0.01
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :] # [num_bb, 4]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            # box_prob [num_bb(gt0.01), 5]
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold, # nms:hardなら使われない(上で既に適用済)
                                      iou_threshold=self.iou_threshold, # nmsのIoUthresh   
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


    def predict_resnet(self, image, top_k=-1, prob_threshold=None):
        """
        image : [h, w, c]
        """
        print(image.shape)
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes, _ = self.net.forward(images) # resnet50_ssd.Res_SSD.forward
            print("Inference time: ", self.timer.end())

        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :] # [num_bb, 4]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            # box_prob [num_bb(gt0.01), 5]
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold, # nms:hardなら使われない(上で既に適用済)
                                      iou_threshold=self.iou_threshold, # nmsのIoUthresh   
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


    # custom function
    def predict_for_genImage(self, image, top_k=200, prob_threshold=None, IoU_thresh=0.45):
        """
        image : [h, w, 3], np
        """
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape # [h, w]
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device) # [1, 3, 300, 300]
        with torch.no_grad():
            # self.timer.start()
            row_conf, row_boxes, _ = self.net.forward(images) # [1, 8732, 21] / [1, 8732, 4]
            
            # print("Inference time: ", self.timer.end())
        #########################################################
        batch, _, num_classes = row_conf.shape[:3]
        output = torch.zeros(batch, num_classes, top_k, 5) # [1, 21, 200, 5]
        selected = np.zeros((batch, num_classes, top_k)) # [1, 21, 200]
        prob_threshold = 0.01 # 固定してしまう
        
        for i in range(batch):
            conf_scores = row_conf[i].clone().permute(1,0) # [21, 8732]

            for cl in range(1, num_classes): # cl = 0 is skipped : background
                c_mask = conf_scores[cl].gt(prob_threshold) # prob_threshold(0.01)より大きい値のbox(8732)を探す, c_mask.shape = [8732]

                scores = conf_scores[cl][c_mask] # [num_bb]
                
                # conf_threshを超えるboxがひとつもない場合：scores.shape >> [0] つまりdim=1, intの0と同じ            
                if scores.shape == torch.Size([0]): # このクラスのthresh gt はない
                    continue
                indicies = np.where(c_mask.data.cpu()==1)[0] # 0.01box
                # print('indicies ', indicies)

                l_mask = c_mask.unsqueeze(1).expand_as(row_boxes[i]) # > [8732, 4] 8732のうちgtだったボックスだけ[1,1,1,1]
                # print(l_mask.shape)
                boxes = row_boxes[i][l_mask].view(-1, 4) # [num_bb, 4]

                # idx of highest scoring and non-overlapping boxes per class
                try:
                    ids, count = nms(boxes, scores, IoU_thresh, top_k) # ids : indicies(~num_bb)のうち何番目のボックスがつかわれたか
                except:
                    print("exception occured.")
                    continue

                # c_maskで8732 -> 0.01boxに絞る(greater than 0.01)
                # nmsで n -> len(ids)に絞る．ids = nから絞った残りboxのindex.
                # --- > idsに入っているindexがそれぞれ8732中のどのboxに対応するのかわかれば追跡可能
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                ###clamp
                output[i, cl, :count, 1:] = torch.clamp(output[i, cl, :count, 1:], min=0.0, max=1.0)
                selected[i, cl, :count] = indicies[ids.data.cpu()[:count]] # selected(~8732)のうち何番目のボックスが使われたか

        flt = output.contiguous().view(batch, -1, 5) # [1,21,200,5] to [1, 4200, 5](conf,x,y,w,h)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        # print(output.shape, output)

        return output, torch.from_numpy(selected), height, width


    # remove with_nograd, and adapt multiple images(batch)
    def predict_for_batchImages(self, images, top_k=200, prob_threshold=None, IoU_thresh=0.45):
        """
        image : [b, h, w, 3], np
        """
        cpu_device = torch.device("cpu")
        _, height, width, _ = images.shape # [h, w]
        
        trans_images = []
        for i in range(len(images)):
            im = self.transform(images[i])
            im = im.unsqueeze(0) # [1, 3, 300, 300]
            trans_images.append(im)
        trans_images = torch.cat(trans_images, dim=0)
        
        images = trans_images.to(self.device) # [1, 3, 300, 300]
        
        row_conf, row_boxes, row_scores= self.net.forward(images) # [b, 8732, 21] / [b, 8732, 4] / [b, 8732, 21]
        #########################################################
        batch, _, num_classes = row_conf.shape[:3]
        output = torch.zeros(batch, num_classes, top_k, 5) # [b, 21, 200, 5]
        selected = np.zeros((batch, num_classes, top_k)) # [b, 21, 200]
        prob_threshold = 0.01 # 固定してしまう
        
        for i in range(batch):
            conf_scores = row_conf[i].clone().permute(1,0) # [21, 8732]

            for cl in range(1, num_classes): # cl = 0 is skipped : background
                c_mask = conf_scores[cl].gt(prob_threshold) # prob_threshold(0.01)より大きい値のbox(8732)を探す, c_mask.shape = [8732]

                scores = conf_scores[cl][c_mask] # [num_bb]
                
                # conf_threshを超えるboxがひとつもない場合：scores.shape >> [0] つまりdim=1, intの0と同じ            
                if scores.shape == torch.Size([0]): # このクラスのthresh gt はない
                    continue
                indicies = np.where(c_mask==1)[0] # 0.01box
                # print('indicies ', indicies)

                l_mask = c_mask.unsqueeze(1).expand_as(row_boxes[i]) # > [8732, 4] 8732のうちgtだったボックスだけ[1,1,1,1]
                # print(l_mask.shape)
                boxes = row_boxes[i][l_mask].view(-1, 4) # [num_bb, 4]
                # idx of highest scoring and non-overlapping boxes per class

                try:
                    ids, count = nms(boxes.data, scores.data, IoU_thresh, top_k) # ids : indicies(~num_bb)のうち何番目のボックスがつかわれたか
                except:
                    print("exception occured.")
                    continue
                # c_maskで8732 -> 0.01boxに絞る(greater than 0.01)
                # nmsで n -> len(ids)に絞る．ids = nから絞った残りboxのindex.
                # --- > idsに入っているindexがそれぞれ8732中のどのboxに対応するのかわかれば追跡可能
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                ###clamp
                output[i, cl, :count, 1:] = torch.clamp(output[i, cl, :count, 1:], min=0.0, max=1.0)

                selected[i, cl, :count] = indicies[ids[:count]] # selected(~8732)のうち何番目のボックスが使われたか

        flt = output.contiguous().view(batch, -1, 5) # [1,21,200,5] to [1, 4200, 5](conf,x,y,w,h)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        # print(output.shape, output)

        return output, torch.from_numpy(selected), height, width, row_scores # row_score : probability


     # for scale variance transform.
    def predict_various(self, images, height, width, top_k=200, prob_threshold=None, IoU_thresh=0.45):
        """
        images : [b(1), h, w, 3], np
        """
        # assert images.shape[-1] == 3
        cpu_device = torch.device("cpu")
        # print()
        # print(images.shape)
        # print()

        # trans_images = []
        # for i in range(len(images)):
        #     im = self.transform(images[i])
        #     im = im.unsqueeze(0) # [1, 3, 300, 300]
        #     trans_images.append(im)
        # trans_images = torch.cat(trans_images, dim=0)

        images = self.transform(images) # [3, 300, 300]
        images = images.unsqueeze(0) # [1, 3, 300, 300]
        # images = trans_images.to(self.device) # [b, 3, 300, 300]
        images = images.to(self.device) # [b, 3, 300, 300]
        print(f'imgs:{images.shape}')

        row_conf, row_boxes, row_scores= self.net.forward(images) # [b, 8732, 21] / [b, 8732, 4] / [b, 8732, 21]
        # print('row_conf************************************* ')
        # print(row_conf)
        #########################################################
        batch, _, num_classes = row_conf.shape[:3]
        output = torch.zeros(batch, num_classes, top_k, 5) # [b, 21, 200, 5]
        selected = np.zeros((batch, num_classes, top_k)) # [b, 21, 200]
        prob_threshold = 0.01 # 固定してしまう
        
        for i in range(batch):
            conf_scores = row_conf[i].clone().permute(1,0) # [21, 8732]

            for cl in range(1, num_classes): # cl = 0 is skipped : background
                c_mask = conf_scores[cl].gt(prob_threshold) # prob_threshold(0.01)より大きい値のbox(8732)を探す, c_mask.shape = [8732]

                scores = conf_scores[cl][c_mask] # [num_bb]
                
                # conf_threshを超えるboxがひとつもない場合：scores.shape >> [0] つまりdim=1, intの0と同じ            
                if scores.shape == torch.Size([0]): # このクラスのthresh gt はない
                    continue
                indicies = np.where(c_mask==1)[0] # 0.01box
                # print('indicies ', indicies)

                l_mask = c_mask.unsqueeze(1).expand_as(row_boxes[i]) # > [8732, 4] 8732のうちgtだったボックスだけ[1,1,1,1]
                # print(l_mask.shape)
                boxes = row_boxes[i][l_mask].view(-1, 4) # [num_bb, 4]
                # idx of highest scoring and non-overlapping boxes per class

                try:
                    ids, count = nms(boxes.data, scores.data, IoU_thresh, top_k) # ids : indicies(~num_bb)のうち何番目のボックスがつかわれたか
                except:
                    print("exception occured.")
                    continue
                # c_maskで8732 -> 0.01boxに絞る(greater than 0.01)
                # nmsで n -> len(ids)に絞る．ids = nから絞った残りboxのindex.
                # --- > idsに入っているindexがそれぞれ8732中のどのboxに対応するのかわかれば追跡可能
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                ###clamp
                output[i, cl, :count, 1:] = torch.clamp(output[i, cl, :count, 1:], min=0.0, max=1.0)

                selected[i, cl, :count] = indicies[ids[:count]] # selected(~8732)のうち何番目のボックスが使われたか

        flt = output.contiguous().view(batch, -1, 5) # [1,21,200,5] to [1, 4200, 5](conf,x,y,w,h)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        # print(output.shape, output)

        return output, torch.from_numpy(selected), height, width, row_scores, row_boxes # row_score : probability




    def predict_various_DAVIS(self, images, height, width, top_k=200, prob_threshold=None, IoU_thresh=0.5):
        """
        images : [b(1), h, w, 3], np
        """
        assert images.shape[1] == 3
        cpu_device = torch.device("cpu")
        images = images.to(self.device) # [b, 3, 300, 300]
        row_conf, row_boxes, row_scores= self.net.forward(images) # [b, 8732, 21] / [b, 8732, 4] / [b, 8732, 21]
        #########################################################
        batch, _, num_classes = row_conf.shape[:3]
        output = torch.zeros(batch, num_classes, top_k, 5) # [b, 21, 200, 5]
        selected = np.zeros((batch, num_classes, top_k)) # [b, 21, 200]
        prob_threshold = 0.01 # 固定してしまう
        
        for i in range(batch):
            conf_scores = row_conf[i].clone().permute(1,0) # [21, 8732]
            
            for cl in range(1, num_classes): # cl = 0 is skipped : background
                print("conf_max: ", conf_scores[cl].max())
               
                c_mask = conf_scores[cl].gt(prob_threshold) # prob_threshold(0.01)より大きい値のbox(8732)を探す, c_mask.shape = [8732]
                scores = conf_scores[cl][c_mask] # [num_bb]
                
                # conf_threshを超えるboxがひとつもない場合：scores.shape >> [0] つまりdim=1, intの0と同じ            
                if scores.shape == torch.Size([0]): # このクラスのthresh gt はない
                    continue
                indicies = np.where(c_mask.data.cpu()==1)[0] # 0.01box
                l_mask = c_mask.unsqueeze(1).expand_as(row_boxes[i]) # [8732, 4] 8732のうちgtだったボックスだけ[1,1,1,1]
                boxes = row_boxes[i][l_mask].view(-1, 4) # [num_bb, 4]
                
                # idx of highest scoring and non-overlapping boxes per class
                try:
                    # print("@@@nms", cl)
                    ids, count = nms(boxes.data.cpu(), scores.data.cpu(), IoU_thresh, top_k) # ids : indicies(~num_bb)のうち何番目のボックスがつかわれたか
                except:
                    print("exception occured.")
                    continue

                # c_maskで8732 -> 0.01boxに絞る(greater than 0.01)
                # nmsで n -> len(ids)に絞る．ids = nから絞った残りboxのindex.
                # --- > idsに入っているindexがそれぞれ8732中のどのboxに対応するのかわかれば追跡可能
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                ###clamp
                output[i, cl, :count, 1:] = torch.clamp(output[i, cl, :count, 1:], min=0.0, max=1.0)
                selected[i, cl, :count] = indicies[ids[:count]] # selected(~8732)のうち何番目のボックスが使われたか
                print()

        flt = output.contiguous().view(batch, -1, 5) # [1,21,200,5] to [1, 4200, 5](conf,x,y,w,h)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output, torch.from_numpy(selected), height, width, row_scores, row_boxes # row_score : probability


    def predict_oneBox(self, images, height, width, top_k=200, prob_threshold=None, IoU_thresh=0.45):
        """
        images : [b, 3, h, w], np
        """
        cpu_device = torch.device("cpu")
        
        images = images.to(self.device) # [b, 3, 300, 300]
        # print(f'imgs:{images.shape} {images[0,0,:5,:5]}')
        row_conf, row_boxes, row_scores = self.net.forward(images) # [b, 8732, 21] / [b, 8732, 4] / [b, 8732, 21]
        # print('row_conf************************************* ')
        # print(row_conf)
        #########################################################
        batch, _, num_classes = row_conf.shape[:3]
        output = torch.zeros(batch, num_classes, top_k, 5) # [b, 21, 200, 5]
        selected = np.zeros((batch, num_classes, top_k)) # [b, 21, 200]
        prob_threshold = 0.01 # 固定してしまう
        
        for i in range(batch):
            conf_scores = row_conf[i].clone().permute(1,0) # [21, 8732]
            for cl in range(1, num_classes): # cl = 0 is skipped : background
                c_mask = conf_scores[cl].gt(prob_threshold) # prob_threshold(0.01)より大きい値のbox(8732)を探す, c_mask.shape = [8732]

                scores = conf_scores[cl][c_mask] # [num_bb]
                # conf_threshを超えるboxがひとつもない場合：scores.shape >> [0] つまりdim=1, intの0と同じ            
                if scores.shape == torch.Size([0]): # このクラスのthresh gt はない
                    continue
                indicies = np.where(c_mask.data.cpu().numpy()==1)[0] # 0.01box

                l_mask = c_mask.unsqueeze(1).expand_as(row_boxes[i]) # > [8732, 4] 8732のうちgtだったボックスだけ[1,1,1,1]
                # print(l_mask.shape)
                boxes = row_boxes[i][l_mask].view(-1, 4) # [num_bb, 4]
                # idx of highest scoring and non-overlapping boxes per class

                try:
                    ids, count = nms(boxes.data, scores.data, IoU_thresh, top_k) # ids : indicies(~num_bb)のうち何番目のボックスがつかわれたか
                except:
                    print("exception occured.")
                    continue
                # c_maskで8732 -> 0.01boxに絞る(greater than 0.01)
                # nmsで n -> len(ids)に絞る．ids = nから絞った残りboxのindex.
                # --- > idsに入っているindexがそれぞれ8732中のどのboxに対応するのかわかれば追跡可能
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                ###clamp
                output[i, cl, :count, 1:] = torch.clamp(output[i, cl, :count, 1:], min=0.0, max=1.0)

                selected[i, cl, :count] = indicies[ids.data.cpu()[:count]] # selected(~8732)のうち何番目のボックスが使われたか

        flt = output.contiguous().view(batch, -1, 5) # [1,21,200,5] to [1, 4200, 5](conf,x,y,w,h)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        # print(output.shape, output)

        return row_conf, row_boxes, row_scores, height, width