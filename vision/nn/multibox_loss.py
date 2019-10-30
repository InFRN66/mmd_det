import torch.nn as nn
import torch.nn.functional as F
import torch
from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations, mask=None):
        """Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions. detection layerのraw出力
            predicted_locations (batch_size, num_priors, 4): predicted locations. detection layerのraw出力
            labels (batch_size, num_priors): real labels of all the priors.　各priorsに割り当てられたtargetのlabelを保持，IoU<0.5(matching_IoU)は全てid=0
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors. 各priorsに割り当てられたgtboxの座標とprior座標のoffset
        """       
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0] # 各priorsのbackground classのLoss
            pos_mask, neg_mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio) # pos+negに使うmask [batch, 8732] = pos_plus_neg_mask
            mask = pos_mask | neg_mask

        confidence = confidence[mask, :] # [pos+neg, 21]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False) # confidenceはまだsoftmaxもlogもとってない
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4) # [batch, 8732, 4] -> [batch*8732, 4]
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4) # [batch, 8732, 4] -> [batch*8732, 4]
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)  # 全batchの合計pos
        return smooth_l1_loss/num_pos, classification_loss/num_pos


# --- 6/25
class MultiboxLoss_WideSamples(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio, center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss_WideSamples, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)


    def forward(self, confidence, predicted_locations, labels, gt_locations, mask=None):
        """Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions. detection layerのraw出力
            predicted_locations (batch_size, num_priors, 4): predicted locations. detection layerのraw出力
            labels (batch_size, num_priors): real labels of all the priors.　各priorsに割り当てられたtargetのlabelを保持，IoU<0.5(matching_IoU)は全てid=0
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors. 各priorsに割り当てられたgtboxの座標とprior座標のoffset
        """
        num_classes = confidence.size(2)
        # with torch.no_grad():
        # --- pos_mask = mask > 0 # [batch, prior] 0/1
        log_softmax = -F.log_softmax(confidence, dim=-1) # [b, prior, class]
        background_loss = log_softmax[:,:, 0]

        # [batch, prior] 1 / 0
        # --- 通常のHNM
        pos_mask, neg_mask = box_utils.hard_negative_mining(background_loss, labels, self.neg_pos_ratio)
        # # --- positive weightでのHNM
        # pos_mask, neg_mask = box_utils.hard_negative_mining_positive_weight(background_loss, labels, mask, self.neg_pos_ratio)
        
        # --- 使用するpos_mask, neg_maskの割合(pos_mask = 個数ではなくweight) 
        masking = mask[pos_mask].reshape(-1) # [num_positive]
        labeling = labels[pos_mask].tolist() # [num_positive]
        pos_loss = log_softmax[pos_mask].reshape(-1, 21)[[i for i in range(pos_mask.sum())], labeling].reshape(-1) # [num_positive]
        confidence_pos_loss = masking * pos_loss
        
        # [num_negative]
        confidence_neg_loss = background_loss[neg_mask].reshape(-1)
        confidence_loss = confidence_pos_loss.sum() + confidence_neg_loss.sum()

        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4) # [num_positive, 4]
        gt_locations = gt_locations[pos_mask,:].reshape(-1, 4)  # [num_positive, 4]
        regression_loss = smooth_l1_loss_sigmoid(predicted_locations, gt_locations, masking)   
        
        # all_batch_pos = pos_mask.sum()  # positive sample sum
        all_batch_pos = mask.sum() # positive weiht sum
        
        return regression_loss / all_batch_pos, confidence_loss / all_batch_pos 


def smooth_l1_loss_sigmoid(x, y, masking):
    """
    Args:
        x: prediction [pos_neg_mask, 4]
        y: ground truth [pos_neg_mask, 4]
        masking : []
    """
    assert masking.shape[0] == x.shape[0] == y.shape[0]
    masking = masking.reshape(-1, 1).expand(-1, 4)
    assert masking.shape == x.shape == y.shape
    z1 = 0.5 * torch.pow(x-y, 2)
    z2 = torch.abs(x-y) - 0.5
    out = torch.where(torch.abs(x - y) < 1, z1, z2)
    out = out * masking  # ここから上はF.smooth_l1_lossと同じ
    return out.sum()
