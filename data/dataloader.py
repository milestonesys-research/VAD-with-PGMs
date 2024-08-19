# MIT License
# 
# Copyright (c) 2024 milestonesys-research
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch

from tqdm import tqdm

from detectron2.structures.boxes import Boxes

import data.var_lib as var_lib


############################
# List of functions
############################

# get_confidence_score_threshold(dataset_name)
# load_objects(dataset_name, data_type, filename_list_sorted, conf_thresh_values=(0.5, 0.5))

############################


def get_confidence_score_threshold(dataset_name):

    root_file_path = var_lib.METADATA_DIR__ROOT.format(dataset_name) #, 'train')
    meta_data = np.loadtxt(root_file_path + 'toy00__gt.txt', dtype=str)

    conf_scores__people = []
    conf_scores__objects = []


    for data in tqdm(meta_data):
        
        # Placeholder values assigned to pred_class and pred_conf
        # for demonstration purpose
        pred_class = 0 # int(data[...])
        pred_conf = 1.0 # np.float32(data[...])
        
        if pred_class == 0:
            conf_scores__people.append(pred_conf)
        else:
            conf_scores__objects.append(pred_conf)


    
    avg_score__people = np.mean(conf_scores__people)
    avg_score__objects = np.mean(conf_scores__objects)
    
    std_score__people = np.std(conf_scores__people)
    std_score__objects = np.std(conf_scores__objects)
    
    print(avg_score__people - (2.0 * std_score__people))
    print(avg_score__objects - (2.0 * std_score__objects))


    return (avg_score__people - (2.0 * std_score__people), avg_score__objects - (2.0 * std_score__objects))


def load_objects(dataset_name, data_type, filename_list_sorted, conf_thresh_values=(0.5, 0.5)):

    root_file_path = var_lib.METADATA_DIR__ROOT.format(dataset_name) #, data_type)
    meta_data = np.loadtxt(root_file_path + 'toy00__gt.txt', dtype=str)
    meta_data_files = [ f[0] for f in meta_data ]

    box_list__all, features_list__all = [], []
    box_list__frame, features_list__frame = [], []
    box_count__all = 0


    for file in tqdm(filename_list_sorted):

        # No objects detected in current frame
        if not file in meta_data_files:
            box_list__all.append(None)
            features_list__all.append(None)
            continue

        data = meta_data[meta_data_files.index(file), :]
        
        # Placeholder values assigned to pred_class and pred_conf
        # for demonstration purpose
        pred_class = 0 # int(data[...])
        pred_conf = 1.0 # np.float32(data[...])

        pred_track_id = np.int32(data[1])

        # Following the MS-COCO convention, humans are labeled with integer 0
        if ((pred_class != 0) and (pred_conf < conf_thresh_values[1])) or ((pred_class == 0) and (pred_conf < conf_thresh_values[0])):
            box_list__all.append(None)
            features_list__all.append(None)
            continue
        

        # Bounding box coordinates appended to box_list__frame are of format [center_x, center_y, width, height]
        x, y, w, h = np.float32(data[2]), np.float32(data[3]), np.float32(data[4]), np.float32(data[5])
        x1, x2 = x - (0.5 * w), x + (0.5 * w)
        y1, y2 = y - (0.5 * h), y + (0.5 * h)
        
        box_list__frame.append([x1,y1,x2,y2])
        features_list__frame.append([pred_class, pred_conf, pred_track_id])



        box_count__all += len(box_list__frame)

        boxes = Boxes(torch.from_numpy(np.array(box_list__frame, dtype=np.float32)))
        box_list__all.append(boxes.clone())
        features_list__all.append(features_list__frame.copy())

        box_list__frame.clear()
        features_list__frame.clear()
        



    return box_list__all, features_list__all, box_count__all


