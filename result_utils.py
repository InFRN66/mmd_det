import numpy as np
import os, sys, json, pickle, csv

def save_result(ID, TYPE, RANGE, Data, PLACE, save_dir):
    """
    for DAVIS
    """
    header = ['voc_id', 'type', 'range', 'track_box_index', 'featuremap', 'anchor', 'grid_y', 'grid_x', 'factor', 'probability']
    with open(os.path.join(save_dir, f'result_{ID}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for track_index in Data.keys():
            data = np.array(Data[track_index])
            data[:, 1] = (data[:,0] * data[:,1]) + 30 # factor * (1 or -1)
            data = data[:, 1:]
            num_points = len(data)

            box = PLACE[track_index]
            box = np.array(box).reshape(1,-1).repeat(num_points, 0)
            voc_id = np.array(ID).repeat(num_points).reshape(-1, 1)
            type_ = np.array(TYPE).repeat(num_points).reshape(-1, 1)
            range_ = np.array(RANGE[0]).repeat(num_points).reshape(-1, 1) # [3, -3]とかのうち正の方だけ使う
            track_index = np.array(track_index).repeat(num_points).reshape(-1, 1)
            csv_body = np.hstack((voc_id, type_, range_, track_index, box, data))
            
            writer.writerows(csv_body)
    return 0


def save_result_kai(obj, TYPE, Data, PLACE, save_dir):
    """
    for VOC random seed
    """
    header = ['type', 'track_box_index', 'featuremap', 'anchor', 'grid_y', 'grid_x', 'factor', 'probability']
    with open(os.path.join(save_dir, f'object{obj}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for track_index in Data.keys():
            data = np.array(Data[track_index])
            data = data[:, 1:]
            num_points = len(data)
            box = PLACE[track_index]
            box = np.array(box).reshape(1,-1).repeat(num_points, 0)
            
            type_ = np.array(TYPE).repeat(num_points).reshape(-1, 1)
            track_index = np.array(track_index).repeat(num_points).reshape(-1, 1)
            csv_body = np.hstack((type_, track_index, box, data))
            
            writer.writerows(csv_body)
    return 0
