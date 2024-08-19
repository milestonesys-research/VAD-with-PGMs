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

import torch
import numpy as np

import tqdm
import copy

from detectron2.structures.boxes import Boxes, pairwise_intersection

import data.var_lib as var_lib


############################
# List of functions
############################
## Helper Functions

# bb_intersection_area(boxA, boxB)
# locate_border(bbox, is_bottom=True)


## Spatial Dimension

# get_bbox_area_cat(bbox, bbox_avg, bbox_std)
# get_bbox_aspect_ratio_cat(bbox)
# get_bbox_cell_intersection_cat(bbox, cell)


## Spatio-Temporal Dimension

# calc_bbox_velocity__displacement(prev_center, curr_center, frame_steps)
# get_velocity_cat(bbox_velocity, v_mean, v_std)
# get_direction_angle(opposite, adjacent)
# get_bbox_moving_direction_cat(prev_center, curr_center)


## Overall Statistics

# get_bbox_stats(train_boxes, train_features)
# get_velocity_stats(frame_steps_temp, train_boxes, train_features)


############################


# Helper Functions

def bb_intersection_area(boxA, boxB):
    area = pairwise_intersection(boxA, boxB)[0][0].to("cpu").numpy()
    return area


def locate_border(bbox, is_bottom=True):
    
    bbox = bbox.tensor.squeeze().numpy()

    x_min, x_max = np.int32(bbox[0]), np.int32(bbox[2])

    y_max = np.int32(bbox[3])
    y_min = np.int32(bbox[1])
    
    # Following visualization convenion in OpenCV
    if is_bottom == True:
        y_min = y_max

    y_min = (y_min - 1) if y_min > 0 else y_min
    y_max = (y_max - 1) if y_max > 0 else y_max
    x_min = (x_min - 1) if x_min > 0 else x_min

    y_min = (var_lib.IMG_HEIGHT - 1) if y_min >= var_lib.IMG_HEIGHT else y_min
    y_max = (var_lib.IMG_HEIGHT - 1) if y_max >= var_lib.IMG_HEIGHT else y_max
    x_max = (var_lib.IMG_WIDTH - 1) if x_max >= var_lib.IMG_WIDTH else x_max
    
    border_x = np.arange(x_min, x_max + 1)
    border = []

    if is_bottom == True:
        for x in border_x:
            border.append(var_lib.GRID_CELL_LOOKUP[y_min, x])
    
    else:
        border_y = np.arange(y_min, y_max + 1)

        for y in border_y:
            for x in border_x:
                border.append(var_lib.GRID_CELL_LOOKUP[y, x])
    
    
    border_cells = list(set(border))
    grid_cells = []

    for cell_id in border_cells:
        anchor_args = np.argwhere(var_lib.GRID_CELL_LOOKUP == cell_id)
        cy_min = np.min(anchor_args[:,0])
        cx_min = np.min(anchor_args[anchor_args[:,0] == cy_min][:,1])
        cy_max = np.max(anchor_args[:,0])
        cx_max = np.max(anchor_args[anchor_args[:,0] == cy_max][:,1])

        grid_cells.append(Boxes(torch.tensor([[cx_min, cy_min, cx_max + 1, cy_max + 1]])))
    




    return border_cells, grid_cells



# Spatial Dimension

def get_bbox_area_cat(bbox, bbox_avg, bbox_std):
    area = np.float32(bbox.area().numpy().squeeze()) / (var_lib.IMG_WIDTH * var_lib.IMG_HEIGHT)

    if area < (bbox_avg - bbox_std):
        return 'xsmall'
    elif area > (bbox_avg + 3 * bbox_std):
        return 'xlarge'
    elif area > (bbox_avg + 2 * bbox_std):
        return 'large'
    elif area > (bbox_avg + bbox_std):
        return 'medium'
    else:
        return 'small'


def get_bbox_aspect_ratio_cat(bbox):
    bbox = bbox.tensor
    aspect_ratio = np.float32(((bbox[:, 2] - bbox[:, 0]) / (bbox[:, 3] - bbox[:, 1])).numpy())

    if aspect_ratio < 1.1:
        return 'portrait'
    elif aspect_ratio > 1.1:
        return 'landscape'
    else:
        return 'square'


def get_bbox_cell_intersection_cat(bbox, cell):
    cell_area_rel = np.float32((bb_intersection_area(cell, bbox) / cell.area()).numpy())
    
    if cell_area_rel == 1.0:
        return 'full'
    elif cell_area_rel >= 0.75:
        return '3/4'
    elif cell_area_rel >= 0.5:
        return '1/2'
    elif cell_area_rel >= 0.25:
        return '1/4'
    elif cell_area_rel >= 0.0:
        return 'little'
    else:
        return 'none'



# Spatio-Temporal Dimension

def calc_bbox_velocity__displacement(prev_center, curr_center, frame_steps):
    distance = np.sqrt(np.square(prev_center[0] - curr_center[0]) + np.square(prev_center[1] - curr_center[1]))
    velocity = np.round(distance / frame_steps, 4)

    return velocity


def get_velocity_cat(bbox_velocity, v_mean, v_std):
    if bbox_velocity > 6*v_mean+ v_std:
        return 'super_fast'
    elif bbox_velocity > 3.5*v_mean + v_std:
        return 'fast'
    else :
        return 'normal'


def get_direction_angle(opposite, adjacent):
    alpha = np.arctan2(opposite, adjacent) * 180 / np.pi

    return alpha


def get_bbox_moving_direction_cat(prev_center, curr_center):
    opposite, adjacent = curr_center[1] - prev_center[1], curr_center[0] - prev_center[0]
    angle = get_direction_angle(opposite, adjacent)

    if (angle >= 67.5) and (angle < 112.5):
        return 'N'
    elif (angle >= 22.5) and (angle < 67.5):
        return 'NE'
    elif (angle >= 112.5) and (angle < 157.5):
        return 'NW'
    elif (angle >= 157.5) and (angle < -157.5):
        return 'W'
    elif (angle >= -157.5) and (angle < 135):
        return 'SW'
    elif (angle >= -112.5) and (angle < -67.5):
        return 'S'
    elif (angle >= -67.5) and (angle < 22.5):
        return 'SE'
    else:
        return 'E'



# Overall Statistics

def get_bbox_stats(train_boxes, train_features):
    """
    Calculate statistics of bounding boxes in the dataset.

    Parameters:
    - train_boxes (list): List of bounding boxes for training data.
    - train_features (list): List of features extracted from training data.

    Returns:
    - bbox_stats_dict (dict): Dictionary containing statistics of bounding boxes.
        Format: {scene_id: {class_int: { 'area_stats': { 'min': min_area, 'max': max_area, 'mean': mean_area, 'std': std_area } } } }
    """
    
    
    num_images = int(len(train_boxes))

    area_stats_dict = {}
    skipped_count = 0
        
    for img_idx in tqdm.tqdm(range(num_images)):

        boxes = train_boxes[img_idx]
        if boxes is None:
            continue

        features = train_features[img_idx]
        
        for bidx in range(len(boxes)):
            bbox, feature = boxes[bidx], features[bidx]
            bbox_area = bbox.area().numpy().squeeze() / (var_lib.IMG_WIDTH * var_lib.IMG_HEIGHT)

            if bbox_area == 0.0:
                skipped_count += 1
                continue
            
            class_int = feature[0]

            prev_areas = area_stats_dict.get(class_int)
            if prev_areas is None:
                area_stats_dict.update({ class_int: [bbox_area] })
            else:
                prev_areas.append(bbox_area)
                area_stats_dict.update({ class_int: prev_areas })
            
        

    print("\n\nSkipped: ", skipped_count)

    bbox_stats_dict = {}

    for class_item in area_stats_dict.items():
        area_dict = { 'area_stats': { 
                                    'min': np.min(class_item[1]),
                                    'max': np.max(class_item[1]),
                                    'mean': np.mean(class_item[1]),
                                    'std': np.std(class_item[1])
                                } }

        
        prev_areas = bbox_stats_dict.get(class_item[0])
        assert prev_areas is None
        bbox_stats_dict.update({ class_item[0]: area_dict })
        

    del area_stats_dict
    assert len(bbox_stats_dict) > 0


    return bbox_stats_dict


def get_velocity_stats(frame_steps_temp, train_boxes, train_features):

    num_images = len(train_boxes)    
    class_velocity_stats = {}

    for img_idx in tqdm.tqdm(range(1, num_images)):

        curr_boxes, prev_boxes = train_boxes[img_idx], train_boxes[img_idx - 1]
        
        if curr_boxes is None or (len(curr_boxes) == 1 and np.float32(curr_boxes[0].area().numpy().squeeze()) == 0.0):
            continue

        curr_features, prev_features = train_features[img_idx], train_features[img_idx - 1]
        prev_track_ids = [] if prev_features is None else [int(prev_feat[2]) for prev_feat in prev_features]


        for idx in range(len(curr_boxes)):

            class_int = int(curr_features[idx][0])
            curr_track_id = int(curr_features[idx][2])

            if curr_track_id in prev_track_ids:
                prev_idx = prev_track_ids.index(curr_track_id)
                c_curr, c_prev = curr_boxes[idx].get_centers().numpy().squeeze(), prev_boxes[prev_idx].get_centers().numpy().squeeze()

                bbox_velocity = calc_bbox_velocity__displacement(c_prev, c_curr, frame_steps_temp)
                              
                prev_tracks = class_velocity_stats.get(class_int)
                if prev_tracks is None:
                    class_velocity_stats.update({ class_int: { curr_track_id: [ bbox_velocity ] } })
                else:
                    prev_velocities = prev_tracks.get(curr_track_id)
                    if prev_velocities is None:
                        prev_tracks.update({ curr_track_id: [ bbox_velocity ] })
                    else:
                        prev_velocities.append(bbox_velocity)
                        prev_tracks.update({ curr_track_id: prev_velocities })
                        


    temp = copy.deepcopy(class_velocity_stats)
    class_velocity_stats.clear()
    class_velos = []

    for class_stats in temp.items():
        for object_stats in class_stats[1].values():
            # Average velocities of individual tracks
            class_velos.append(np.mean(object_stats))

        prev_velocities = class_velocity_stats.get(class_stats[0])
        assert prev_velocities is None
        # Velocity statistics over entire class
        class_velocity_stats.update({ class_stats[0]: [np.mean(class_velos), np.std(class_velos), np.min(class_velos), np.max(class_velos)] }) 
            
        class_velos.clear()
    

    assert len(class_velocity_stats) > 0
    
    

    return class_velocity_stats



