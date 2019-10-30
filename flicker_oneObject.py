import numpy as np
import torch
import cv2
from collections import defaultdict
from track_utils import IoU, VOC_CLASSES, COCO_CLASSES
import copy

class Frame:
    def __init__(self, num_scale, num_bb, num_classes, top_k, h, w, c):
        """
        num_scale   : (int) 総フレーム数 = 微小変化で生成した画像群の総数
        num_bb      : (int) 1, 追跡する物体クラスの総数
        num_classes : (int) 21, voc class
        top_k       : (int) 200, 最大検出物体数
        h, w, c     : (int) image size        
        """
        self.h = h
        self.w = w
        self.num_scale = num_scale
        self.num_bb = num_bb
        self.output = torch.zeros(num_scale, num_classes, num_bb, top_k, 5) # 各クラスのdetection結果: 5 = prob, x1, y1, x2, y2

        self.track_box = torch.zeros(num_scale, num_classes, num_bb, 5) # [50, 21, 1, 5]
        self.trajectory = torch.zeros(num_scale, num_bb) # [何番目のimageか，何番目のオブジェクトか]
        self.occlusion_status = torch.ones(num_scale, num_classes) # default one
        self.written_image = torch.zeros(num_scale, h, w, c) # [frame, h, w, c]
        self.box_hide_status = torch.ones(num_scale, num_bb) # num_scale, num_target, 
        # # label操作
        # self.annotation = annotation
        # self._annotation2dets()


    def class_act(self, scale, c, idx, dets):
        '''
        classの処理が終わる度に実行
            self.outputにdetectしたboxの結果
            self.num_bbにdetectしたboxの総数
        '''
        self.output[scale, c, idx, :dets.shape[0], :] = dets # [scale, class, pred_bb, 5] = [batch, 21, ]


    def frame_act(self, idx, written_image):
        '''
        frame処理が終わる度に実行
            self.written_imageにrectangleを書き込んだimg
        '''
        self.written_image[idx, :, :, :] = torch.FloatTensor(written_image)


    def compare_IoU_one(self, current_scale, track_Id, idx, dets, true_box, IoU_thresh=0.5):
        '''
        dets : [numbb, 5] first is conf
        ture_box : [1, 4]
        今フレーム，trackingするクラスのボックスから，最もlabelboxに合うボックスを選定.
        '''
        IoU_store = list()
        # track_box = torch.zeros_like(current_scale, track_Id, len(true_box), 4)
        
        # 今フレームで出ている全ボックスをlabelボックスと比較．
        IoU_store = [ IoU(dets[i, 1:], true_box[0]) for i in range(len(dets)) ] # detect_boxの中で一番truebox近いボックスを探す
        print("predicted dets: {}".format(dets[:3, :]))
        print("true coordinates: {}".format(true_box))
        print("IoU: {}".format(IoU_store))
        IoU_store = torch.FloatTensor(IoU_store)                                                        # 選別にIoU + probabilityを使うのも有りかも
        # しきい値を超えるIoUのboxがあるか．mask
        IoU_mask = IoU_store.ge(IoU_thresh) # [IoU] >= [thresh_IoU] or not
        res_IoU_store = IoU_store * IoU_mask.type(torch.FloatTensor)

        IoU_v, indicies = torch.sort(res_IoU_store, descending=True)
        
        # IoUを満たすboxがない :
        if IoU_mask.sum() == 0:
            print('IoUを満たすboxがない')
            self.trajectory[current_scale, idx] = 1
            # track_box[current_scale, track_Id, b, :5] = track_box[current_scale-1, track_Id, b, :5].clone # 前のスケールと同じボックスにしておく
            self.track_box[current_scale, track_Id, idx, 0] = 0.0
            self.track_box[current_scale, track_Id, idx, 1:] *= 0.0 # detectできないので0にしておく
            return 0, 0, 0
        # IoUを満たすboxがある:
        else:
            print('正解に最も近いboxは IoU={:.4f}, {}/{}番目.'.format(IoU_v[0], indicies[0]+1, len(IoU_store)))
            # 既に1が入っている場合：一回しきい値を下げているのでflickerのままにする．
            self.trajectory[current_scale, idx] = max(0, self.trajectory[current_scale, idx])
            print("1: {}".format(self.trajectory[current_scale, idx]))
            self.track_box[current_scale, track_Id, idx, 0] = self.output[current_scale, track_Id, idx, indicies[0], 0] # prob
            print("2: {}".format(self.trajectory[current_scale, idx]))
            self.track_box[current_scale, track_Id, idx, 1:] = self.output[current_scale, track_Id, idx, indicies[0], 1:] # coods
            print("stored output : {}".format(self.track_box[current_scale, track_Id, idx]))
            print('選ばれたindex =', indicies[0])
            return 1, 1, indicies[0].type(torch.LongTensor)


    def no_detection(self, current_frame, track_Id, idx):
        '''
        detectするboxが一つも見つからなかった場合
        '''        
        # print('no box was detected.')
        # 取り敢えずflickerとして保留
        self.trajectory[current_frame, idx] = 1
        return 0, 0, 0 # 見つからなかったので0を返して，conf_threshを下げてもう一度同じ画像に対しdetectする処理．


    def filter_highprob_trajectory(self, track_Id, conf_thresh):
        '''
        probabilityの流れから，一回でも0.9を超えたもののなかでdetectできなくなったボックスを検出．
        '''
        probability = self.track_box[:, track_Id, 0, 0].cpu().numpy()
        new_status = np.zeros_like(probability)
        high_or_not = (probability>=0.9) # 1 : 0.9以上 / 0 : 0.9より小さい mask
        if high_or_not.sum() == 0:
            return new_status # all 0 で返す
        else:
            new_status = (probability<conf_thresh)
            return new_status # probが低いところを1にして返す


    def box_out_of_area(self, current_frame, idx, gt_box):
        """
        gt_boxが画像枠内の外にある（映っていない）かどうかをself.box_hide_statusに保存
        gt_box: [1, 4], original scale
        """
        # x2 + x1 / 2
        cx = int((gt_box[0, 2] + gt_box[0, 0]) / 2)
        cy = int((gt_box[0, 3] + gt_box[0, 1]) / 2)
        # ボックスが半分以上画像枠外ならstatus1にする
        if cx <= 0 or cx >= self.w or cy <= 0 or cy >= self.h:
            self.box_hide_status[current_frame, idx] = 0 # 0で枠外の意
    

    def fragment_candidate(self, trajectory):
        """
        Args:
            trajectory: [num_scales] 
        各フレームについて，その周囲４フレーム（前後２フレーム）のtrajectoryを見る
        trajectoryは1ならfragment
        周囲４フレームの平均値が0.25以下なら，3/4が正しくdetectされているので周囲の確率は十分高いと言える．
        """
        marginal_fragment_candidate = np.zeros_like(trajectory) # num_scales
        for i in range(len(trajectory)):
            if i >= 2 and i <= len(trajectory)-3:
                length = 4
                mean = (trajectory[i-2] + trajectory[i-1] + trajectory[i+1] + trajectory[i+2]) / length
            elif i == 0:
                length = 2
                mean = (trajectory[i+1] + trajectory[i+2]) / length
            elif i == 1:
                length = 3
                mean = (trajectory[i-1] + trajectory[i+1] + trajectory[i+2]) / length
            elif i == len(trajectory)-2:
                length = 3
                mean = (trajectory[i-2] + trajectory[i-1] + trajectory[i+1]) / length
            elif i == len(trajectory)-1:
                length = 2
                mean = (trajectory[i-2] + trajectory[i-1]) / length
            else:
                raise Exception('invalid index')

            if mean <= 0.25: # 周囲に安定フレーム（0）が3/4以上ならそのフレームは安定
                marginal_fragment_candidate[i] = 1 # 1ならfragmentになりうる資格がある
        return marginal_fragment_candidate # 0:不安定安定なのでfragmentにしたくない（そもそも性能低い） 1:安定 


    def analyse(self, prob_diff, frame_status, track_Id, prob_thresh):
        '''
        prob_status : probのエッジ, prob_thresh(0.4なら40%)より大きい場所が変化率大とする．
        frame_status : trajectory, 単純に一発目でdetectできたか出来ていないか
        '''
        prob_binary = np.zeros_like(prob_diff)
        diff_indicies = np.where(prob_diff >= prob_thresh)[-1]  #差が大きかったフレーム抽出
        prob_binary[diff_indicies] = 1
        
        print("prob_binary")       
        for i in range(len(prob_binary)):
            print("{}: {}".format(i, prob_binary[i]))

        print('prob_diff :', prob_diff)        
        # [ prob_binary, frame_status ]両方１になっていないと検出されない
        prob_binary = prob_binary * frame_status
        print(f'multyply: {prob_binary}')
        print()
        return prob_binary


    # def analyse_probabiity(self, track_Id):
    #     '''
    #     各ボックスの確率 :
    #     '''
    #     print()
    #     probability = self.track_box[:, track_Id, 0, 0].cpu().numpy()
    #     F = np.array([-1,1]) # フィルタ
        
    #     # スタート拡張
    #     c_probability = probability[:]
    #     c_probability = np.hstack((np.repeat(c_probability[0], 1), c_probability))
    #     # c_probability = np.hstack((c_probability, np.repeat(c_probability[-1], 1)))

    #     print('prob', np.array(probability))

    #     new = np.zeros_like(probability)
    #     for i, idx in enumerate(range(len(probability))):
    #         idx += 1
    #         new[i] += (F * c_probability[idx-1:idx+1]).sum()
    #         # print(new)
    #     print(new)
    #     print()
    #     new = np.minimum(new, 0) # 大 -> 小になるものだけ取り出す
    #     return new


    def analyse_probabiity_percent(self, track_Id, frame_status, obj, adjacent_thresh):
        '''
        前フレームとのprobの差分を調べる / 前フレームのprobを１として後フレームがどれだけ下がったか
        thresh 
        '''
        probability = self.track_box[:, track_Id, obj, 0].cpu().numpy()
        print(f'probability in this object')  # shape = 総フレーム数
        for i in range(len(probability)):
            print("{}: {}".format(i, probability[i]))
        F = np.array([1,-1]) # フィルタ
        
        # スタート拡張
        c_probability = probability[:]
        c_probability = np.hstack((np.repeat(c_probability[0], 1), c_probability))
        # c_probability = np.hstack((c_probability, np.repeat(c_probability[-1], 1)))

        new = np.zeros_like(probability)
        for i, idx in enumerate(range(len(probability))):
            idx += 1
            new[i] += (F * c_probability[idx-1:idx+1]).sum() / (c_probability[idx-1]+1e-3)
        new = np.maximum(new, 0) # 大 -> 小になる（下がる）ものだけ取り出す

        # 前後１フレーム両方で出力が大きい(adjacent_thresh以上)：　以外の場合はfragmentとして出力しない
        for j in range(1, len(new)-1):
            if j == len(new)-2:
                new[-1] = 0 # 最後のフレームは検出されないように
                continue
            else:
                if np.array(probability)[j-1] < adjacent_thresh or np.array(probability)[j+1] < adjacent_thresh: # or なのは対偶取ってるから
                    new[j] = 0 # 前フレームとの差分を0とする(errorとして検出されないようにする)
        return np.abs(new), np.array(probability)


    def add_flickSign(self, f, status):
        # flicker
        if status == 1:
            new = cv2.circle(self.written_image[f].cpu().numpy(), (30,30), 20, (255,0,0), -1) # red
            self.written_image[f] = torch.FloatTensor(new)
        # occlusion
        elif status == -1:
            new = cv2.circle(self.written_image[f].cpu().numpy(), (30,30), 20, (0,0,255), -1) # blue
            self.written_image[f] = torch.FloatTensor(new)
        elif status == 2:
            new = cv2.circle(self.written_image[f].cpu().numpy(), (30,30), 20, (0,128,0), -1) # green
            self.written_image[f] = torch.FloatTensor(new)

############################################################################################################
