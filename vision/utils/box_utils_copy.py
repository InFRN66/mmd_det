import collections
import torch
import itertools
from typing import List
import math
from collections import namedtuple, defaultdict

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            # ['feature_map_size', 'shrinkage', 'box_sizes(min, max)', 'aspect_ratios']
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]), 大きい正方形は小さいスケール*sqrt(60/30)
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]), 大きい正方形は小さいスケール*sqrt(111/60)
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]), 大きい正方形は小さいスケール*sqrt(162/111)
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2]) 大きい正方形は小さいスケール*sqrt(315/264)
            ]
        image_size: image size, (= 300)
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage # 300 / 8 = 37.5, 300 / 16 = 18.75, ...
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale # (0+0.5)/38, (1+0.5)/38, ...
            y_center = (j + 0.5) / scale

            # small sized square box (1)
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # big sized square box (1)
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box (2 or 4)
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio) # rout2, rout3
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)

    return priors



# --------------------------------------------------------------------------------

def generate_ssd_priors_custom(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            # ['feature_map_size', 'shrinkage', 'box_sizes(min, max)', 'aspect_ratios']
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]), 大きい正方形は小さいスケール*sqrt(60/30)
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]), 大きい正方形は小さいスケール*sqrt(111/60)
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]), 大きい正方形は小さいスケール*sqrt(162/111)
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2]) 大きい正方形は小さいスケール*sqrt(315/264)
            ]
        image_size: image size, (= 300)
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage # 300 / 8 = 37.5, 300 / 16 = 18.75, ...
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale # (0+0.5)/38, (1+0.5)/38, ...
            y_center = (j + 0.5) / scale

            # small sized square box (1)
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # # big sized square box (1)
            # size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            # h = w = size / image_size
            # priors.append([
            #     x_center,
            #     y_center,
            #     w,
            #     h
            # ])

            # change h/w ratio of the small sized box (2 or 4)
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio) # rout2, rout3
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)

    return priors

# --------------------------------------------------------------------------------

def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    # print('convert area :')
    # print(locations.shape, priors.shape)
    
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2], # real_cebter = ~~
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:] # real_hw = ~~
    ], dim=locations.dim() - 1)



def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    """
    Args:
        center_form_boxes [8732, 4] / (cxcywh)形式，各priorが最も近いと判断したtargetのgtbox情報を保持している(01scale)
        center_form_priors [8732, 4] / (cxcywh)形式，各priorの位置情報（01scale）
    """
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)



def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (b, num_targets, 4): ground truth boxes.
        boxes1 (8732, b, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.01scale
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors * num_targets [8732, num_targets]
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1)) # [8732, num_targets]各gt,dafaultboxのIoU
    # print('ious: ', ious.shape, ious)
    # size: num_priors　どのgtがそのdbに最も合っていたか
    best_target_per_prior, best_target_per_prior_index = ious.max(1) # 8732
    # size: num_targets どのdbがそのgtに最も合っていたか
    best_prior_per_target, best_prior_per_target_index = ious.max(0) # num_targets
    # print('bast matching: ', best_prior_per_target, best_prior_per_target_index)

    for target_index, prior_index in enumerate(best_prior_per_target_index): # this prior is the best for each object. 
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2) # 選ばれたpriorはiouを高くしておく（２）
    # size: num_priors
    # 例；gt_labels[a,b,c], best_target_per_prior_index = [0,0,1,1,1,2,1,0]なら[a,a,b,b,b,c,b,a]a,b,c,はクラスindex
    labels = gt_labels[best_target_per_prior_index] 
    
    # print('max IoU in all priors: ', best_target_per_prior.shape, best_target_per_prior.max())
    labels[best_target_per_prior < iou_threshold] = 0  # IoU以下のpriorはバックグラウンドクラスにする
    boxes = gt_boxes[best_target_per_prior_index]
    print()
    '''
    gt_boxes[best_target_pewr_prior_index] : 
        gt_boxes : [targets, 4] / best_target_per_prior_index : [8732]
        best_target_per_prior_index is in (0 ~ targets-1)
        boxes[idx] = gt_boxes[best_target_per_prior_index] == [8732, 4]
        boxes's row is the gt_boxes which corresponds target.
        boxesにはどれかのtargetのgt_boxの値が入る.
    '''
    return boxes, labels # [8732, 4], [8732]


def assign_priors_adjust_with_anchor(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
    """Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.01scale
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    print(f'gt_labels:{gt_labels}, gt_boxes:{gt_boxes}')
    # size: num_priors * num_targets [8732, num_targets]
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1)) # [8732, num_targets]各gt,dafaultboxのIoU
    # size: num_priors　どのgtがそのdbに最も合っていたか
    best_target_per_prior, best_target_per_prior_index = ious.max(1) # 8732
    # size: num_targets どのdbがそのgtに最も合っていたか
    best_prior_per_target, best_prior_per_target_index = ious.max(0) # num_targets
    which_anchor = discriminate_best_match_anchor(best_prior_per_target_index) # 選択されたtopのアンカーがどのアスペクト比のボックスか
    
    print('bast matching: ', best_prior_per_target, best_prior_per_target_index)
    print('which_anchor: ', which_anchor)

    for target_index, prior_index in enumerate(best_prior_per_target_index): # this prior is the best for each object. 
        best_target_per_prior_index[prior_index] = target_index # targetから見たanchorは何がベストかを優先

    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2) # 選ばれたpriorはiouを高くしておく（２）
    
     
    # 今のtargetに冠する情報のみ抜き出す
    NUM = 0
    boxes, labels = list(), list()
    for idx, uni in enumerate(best_target_per_prior_index.unique().sort()[0]): # target分
        print(f'idx:{idx}, uni:{uni}')
        mask = (best_target_per_prior_index == uni).type(torch.LongTensor) # 8732 // 8732の内，今見ているtargetのindexは1,ほかは0
        NUM += mask.sum()
        # size: num_priors
        # 例；gt_labels[a,b,c], best_target_per_prior_index = [0,0,1,1,1,2,1,0] なら [a,a,b,b,b,c,b,a] a,b,c,はクラスindex, // labels = [b,c,a]
        label = gt_labels[best_target_per_prior_index] * mask # 
        print(f'label.unique():{label.unique()}') # ここで0なのはmaskで削られたということ；つまり今見たいtargetではない

        if which_anchor[uni] in [2,3,4,5]:
            print('anchor expand')
            iou_threshold *= 0.75

        label[best_target_per_prior < iou_threshold] = 0 # IoU以下のpriorはバックグラウンドクラスにする
        box = gt_boxes[best_target_per_prior_index] * (mask.repeat(4,1).t()).type(torch.FloatTensor)

        if idx == 0:
            boxes = box.clone()
            labels = label.clone()
        else:
            boxes += box
            labels += label

    print(f'NUM: {NUM}')
    print(f'boxes.shape{boxes.shape}, labels.shape{labels.shape}')
    print()
    return boxes, labels # [8732, 4], [8732]



def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (batch, num_priors): the loss for each example.
        labels (batch, num_priors): the labels. / 各priorへ割り当てられたtargetのclassid, IoU<0.5のpriorは0になっている．
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    if labels.dim != 2:
        labels = labels.unsqueeze(0)

    # print('loss :', loss.shape, labels.shape, neg_pos_ratio)
    pos_mask = labels > 0 # gtboxとIoU>0.5でマッチしたpriorsだけ抽出するmask [8732]
    num_pos = pos_mask.long().sum(dim=1, keepdim=True) # 各バッチのpositive(IoU>0.5)priorsの合計個数 [batch, 1]
    num_neg = num_pos * neg_pos_ratio # それの何倍か

    # print('posmask', pos_mask.shape, pos_mask)
    loss[pos_mask] = -math.inf # poistiveサンプルのpriorsのbackground lossは-infで極限まで小さく
    # print('loss :', loss.shape, loss)
    _, indexes = loss.sort(dim=1, descending=True) #  ERROR : merge_sort: failed to synchronize: an illegal memory access was encountered
                                                   #  https://github.com/qfgaohao/pytorch-ssd/issues/9 / 大きい順にsort
    _, orders = indexes.sort(dim=1) # 小さい順
    neg_mask = orders < num_neg
    return pos_mask | neg_mask # backgroundのLossが大きい,つまり(backと認識してほしいにも関わらず)backの確率が低いpriorから順に，個数が3倍になるまで取る(or | )


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1) 


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
         boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.[x1,y1,x2,y2, prob]
        iou_threshold: intersection over union threshold. = 0.5
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    
    box_scoresはすでにconfidence_threshで切られている
    まずconfidenceが大きい順に並べ替え，頭から走査
    今調べている（頭にあった）ボックスについて，そのボックス以外の全ボックスとIoU比較，しきい値以上のボックスは全て排除
    排除されなかった残りボックス内で再度confidence最大のボックスを選び・・以下同じ処理
    最終的にボックスが全て排除されるまで行う
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])


def discriminate_best_match_anchor(best_anchors_index):
    '''
    decide the anchor type (0, 1, 2, ...) from index in e.g. 8732
    various in backbone type
    Args:
        best_anchors_index: 各targetでどのanchor(default box)が一番合っているかのindex
    '''
    best_anchors_index = best_anchors_index.clone()

    fm_anchor = namedtuple('Data', ('fm', 'anchor'))
    box_arch = [
        fm_anchor(38, 4),
        fm_anchor(19, 6),
        fm_anchor(10, 6),
        fm_anchor(5, 6),
        fm_anchor(3, 4),
        fm_anchor(1, 4)
    ]

    steps = list(map(lambda x: (x.fm**2)*x.anchor, box_arch))
    steps_ = [ sum(steps[:i]) for i in range(len(steps))] # [0, 5776, 7942, 8542, 8692, 8728]
    
    which_stage = list()
    num_anchor = list()
    print(f'best_anchors_index: {best_anchors_index}')
    for i in best_anchors_index:
        ind = sum([ i > x for x in steps_ ])-1
        print(f'ind: {ind}')
        which_stage.append(ind)
        num_anchor.append(box_arch[ind].anchor)
    in_stage = best_anchors_index.type(torch.FloatTensor) -  torch.Tensor([steps_[which_stage[i]] for i in range(len(best_anchors_index))]).type(torch.FloatTensor)    
    which_anchor = in_stage % torch.Tensor(num_anchor).type(torch.FloatTensor)

    return which_anchor

    