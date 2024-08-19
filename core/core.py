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
from pgmpy.inference import VariableElimination

import tqdm
import time


from core.discrete import *


############################
# List of functions
############################

## Spatial Dimension

# generate_spatial_observations(bbox_stats, train_boxes, train_features, bottom_border_only)
# get_spatial_frame_level_accuracies(model, test_boxes, test_features, train_bbox_stats, bottom_border_only, file_list, save_dir)
# compute_spatial_object_anomaly_score(model, box, feature, bbox_stats, bottom_border_only, filename)


## Spatio-Temporal Dimension

# generate_spatio_temporal_observations(frame_steps_temp, bbox_stats, velo_stats, train_boxes, train_features, bottom_border_only)
# get_spatio_temporal_frame_level_accuracies(model, frame_steps_temp, test_boxes, test_features, train_bbox_stats, train_velo_stats, bottom_border_only, file_list, save_dir)
# compute_spatio_temporal_object_anomaly_score(model, frame_steps_temp, prev_box, curr_box, curr_feature, train_bbox_stats, train_velo_stats, bottom_border_only, filename)


# explain_anomaly_score(anomaly_infer, evidence_dict, class_int, filename, class_prob)


############################



# Spatial Dimension

def generate_spatial_observations(bbox_stats, train_boxes, train_features, bottom_border_only):
    print("\nGenerating observations from training data...")

    coco_names = np.load("./data/coco_names.npy")
    num_images = len(train_boxes)
    
    observation = []
    observations_all = []


    for img_idx in tqdm.tqdm(range(num_images)):

        boxes = train_boxes[img_idx]
        if boxes is None:
            continue

        num_boxes_in_frame = len(boxes)
        if num_boxes_in_frame == 0 or np.all(np.float32(boxes.area().numpy().squeeze()) == 0.0):
            continue


        features = train_features[img_idx]

        for idx in range(len(boxes)):
            bbox, feature = boxes[idx], features[idx]
            class_int = feature[0]
            cell_ids, cell_boxes = locate_border(bbox, bottom_border_only)

            for cidx in range(len(cell_ids)):

                cell = cell_ids[cidx]
                grid_cell = cell_boxes[cidx]
                
                class_stats = bbox_stats.get(class_int)
                assert class_stats is not None, "Error retrieving bbox_stats in generate_spatial_observations()"

                observation.append(0) # Can be altered to account for multiple scenes
                observation.append(cell)             
                observation.append(coco_names[class_int])
                    
                bbox_area_stats = class_stats.get('area_stats')
                bbox_area_mean, bbox_area_std = bbox_area_stats.get('mean'), bbox_area_stats.get('std')
                observation.append(get_bbox_area_cat(bbox, bbox_area_mean, bbox_area_std))
                observation.append(get_bbox_aspect_ratio_cat(bbox))
                observation.append(get_bbox_cell_intersection_cat(bbox, grid_cell[0]))



                observations_all.append(observation.copy())
                observation.clear()
            
        

        
    return observations_all


def get_spatial_frame_level_accuracies(model, test_boxes, test_features, train_bbox_stats, bottom_border_only, file_list, save_dir, explain_anomalies):
    
    coco_names = np.load("./data/coco_names.npy")
    num_images = len(test_boxes)
    frame_scores_dict__all = {}

    for img_idx in tqdm.tqdm(range(num_images)):
        
        filename = file_list[img_idx]

        boxes = test_boxes[img_idx]
        if boxes is None:
            continue

        num_boxes_in_frame = len(boxes)

        if num_boxes_in_frame == 0 or np.all(np.float32(boxes.area().numpy().squeeze()) == 0.0):
            continue


        features = test_features[img_idx]
        object_scores, object_classes, object_confs = [], [], []

        for idx in range(len(boxes)):
            
            if np.float32(boxes[idx].area().numpy().squeeze()) == 0.0:
                object_scores.append(-1.0)
                object_classes.append(None)
                object_confs.append(None)
                continue


            score = compute_spatial_object_anomaly_score(model, boxes[idx], features[idx], train_bbox_stats, bottom_border_only, filename, explain_anomalies)
            
            class_int = features[idx][0]
            class_name = coco_names[class_int]
            conf = np.float16(features[idx][1])

            object_scores.append(score)
            object_classes.append(class_name)
            object_confs.append(conf)


        frame_scores_dict = {}

        if len(object_scores) == 0:
            frame_scores_dict[filename] = [0, None, None, None, None]

        else:

            for i in range(len(object_scores)):
                if object_scores[i] == -1.0:
                    continue

                score = object_scores[i]
                obj_class = object_classes[i]
                obj_conf = object_confs[i]
                bbox = boxes.tensor[i].numpy()

                prev_results = frame_scores_dict.get(filename)
                if prev_results is None:
                    frame_scores_dict[filename] = [1, [score], [obj_class], [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]], [obj_conf]]
                else:
                    prev_results[0] += 1
                    prev_results[1].append(score)
                    prev_results[2].append(obj_class)
                    prev_results[3].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    prev_results[4].append(obj_conf)
                    frame_scores_dict.update({ filename: prev_results })
        
        frame_scores_dict__all.update(frame_scores_dict)

        with open(save_dir, 'a') as f:
            for item in frame_scores_dict.items():
                f.write(str(item) + '\n')
        

    return frame_scores_dict__all


def compute_spatial_object_anomaly_score(model, box, feature, bbox_stats, bottom_border_only, filename, explain_anomalies):
    coco_names = np.load("./data/coco_names.npy")
    class_int = feature[0]
    class_name = coco_names[class_int]

    class_stats = bbox_stats.get(class_int)
    if class_stats is None: # Detected object class has not been encountered during training
        return 0.0
    
    bbox_area_stats = class_stats.get('area_stats')
    bbox_area_mean, bbox_area_std = bbox_area_stats.get('mean'), bbox_area_stats.get('std')

    box_size = get_bbox_area_cat(box, bbox_area_mean, bbox_area_std)
    box_aspect_ratio = get_bbox_aspect_ratio_cat(box)


    anomaly_infer = VariableElimination(model) # BeliefPropagation(model) 
    object_score = 0.0
    
    cell_ids, cell_boxes = locate_border(box, bottom_border_only)
    assert len(cell_ids) > 0
    
    for i in range(len(cell_ids)):
        cell = int(cell_ids[i])
        grid_cell = cell_boxes[i]
        intersect_cat = get_bbox_cell_intersection_cat(box, grid_cell[0])

        evidence_dict = {
            "Scene": 0, # Can be altered to account for multiple scenes
            "GridCell": cell,
            "BoxSize": box_size,
            "BoxAspectRatio": box_aspect_ratio,
            "Intersection": intersect_cat
        }
        

        class_prob = None
        # print("\n\nPredicted: ", class_name)
        # print("Conditional probability table at cell: ", str(covered_cells[i]))
        # print("")

        try:
            cpdt = anomaly_infer.query(variables=["Class"], evidence=evidence_dict, show_progress=False)
            object_scores = cpdt.values
            var_states = cpdt.state_names
            # print(cpdt)

        except:
            # print("\n")
            # print("Issue in image " + str(img_idx) + ":")
            # print("One or more categories not known as evidence:")
            # print(evidence_dict)
            # print("Recording probability score 0.0 for class " + class_name + " in cell " + str(cell) + ".")
            # print("\n")
            class_prob = 0.0
            
        
        if class_prob is None:
            try:
                class_prob = object_scores[var_states['Class'].index(class_name)]
            except:
                # print("\n")
                # print("Issue in image " + str(img_idx) + ":")
                # print("Category '" + class_name + "' not found in probability table.")
                # print("Recording probability score 0.0 for class " + class_name + " in cell " + str(cell) + ".")
                # print("\n")
                class_prob = 0.0

        
        assert class_prob is not None

        if np.isnan(class_prob):
            class_prob = 0.0
        
        object_score += class_prob

        if explain_anomalies:
            explain_anomaly_score(anomaly_infer, evidence_dict, class_int, filename, class_prob)
    

    object_score /= (i + 1)
    assert object_score <= 1.0 # Probabilities must not be greater than 1


    return object_score



# Spatio-Temporal Dimension

def generate_spatio_temporal_observations(frame_steps_temp, bbox_stats, velo_stats, train_boxes, train_features, bottom_border_only):
    print("\nGenerating observations from training data...")


    coco_names = np.load("./data/coco_names.npy")
    num_images = len(train_boxes)
    
    observation = []
    observations_all = []

    for img_idx in tqdm.tqdm(range(1, num_images)):

        curr_boxes, prev_boxes = train_boxes[img_idx], train_boxes[img_idx - 1]
        if (curr_boxes is None):
            continue

        num_boxes_in_frame = len(curr_boxes)
        if num_boxes_in_frame == 0 or np.all(np.float32(curr_boxes.area().numpy().squeeze()) == 0.0):
            continue

        curr_features, prev_features = train_features[img_idx], train_features[img_idx - 1]
        prev_track_ids = [] if prev_features is None else [int(prev_feat[2]) for prev_feat in prev_features]
        
        curr_velocities, curr_directions = [], []


        # Iterate over all boxes first to determine movement velocity and direction
        for idx in range(len(curr_boxes)):
            curr_track_id = int(curr_features[idx][2])

            if curr_track_id in prev_track_ids:
                prev_idx = prev_track_ids.index(curr_track_id)
                c_curr, c_prev = curr_boxes[idx].get_centers().numpy().squeeze(), prev_boxes[prev_idx].get_centers().numpy().squeeze()
                bbox_velocity = calc_bbox_velocity__displacement(c_prev, c_curr, frame_steps_temp)

                class_int = curr_features[idx][0]

                cells, _ = locate_border(curr_boxes[idx], bottom_border_only)
                v_mean, v_std = [], []
                for cell in cells:
                    class_velocity_stats = velo_stats.get(class_int)
                    assert class_velocity_stats is not None, "Error retrieving class_velo_stats in generate_spatio_temporal_observations()"
                    v_mean.append(class_velocity_stats[0])
                    v_std.append(class_velocity_stats[1])
                
                bbox_velocity_cat = get_velocity_cat(bbox_velocity, np.mean(v_mean), np.mean(v_std))
                curr_velocities.append(bbox_velocity_cat)
                
                if bbox_velocity_cat == 'idle':
                    curr_directions.append('none')
                else:
                    object_movement_direction_cat = get_bbox_moving_direction_cat(c_prev, c_curr)
                    curr_directions.append(object_movement_direction_cat)
            
            else:
                curr_velocities.append('undefined')
                curr_directions.append('undefined')
        

        # Iterate over all boxes once more to generate the actual observations
        for idx in range(len(curr_boxes)):
            bbox, feature = curr_boxes[idx], curr_features[idx]
            class_int = feature[0]

            cell_ids, cell_boxes = locate_border(bbox, bottom_border_only)

            for cidx in range(len(cell_ids)):

                cell = cell_ids[cidx]
                grid_cell = cell_boxes[cidx]
                
                class_stats = bbox_stats.get(class_int)
                assert class_stats is not None, "Error retrieving bbox_stats in generate_spatial_observations()"

                observation.append(0) # Can be altered to account for multiple scenes
                observation.append(cell)
                observation.append(coco_names[class_int])
                    
                bbox_area_stats = class_stats.get('area_stats')
                bbox_area_mean, bbox_area_std = bbox_area_stats.get('mean'), bbox_area_stats.get('std')
                observation.append(get_bbox_area_cat(bbox, bbox_area_mean, bbox_area_std))
                observation.append(get_bbox_aspect_ratio_cat(bbox))
                observation.append(get_bbox_cell_intersection_cat(bbox, grid_cell[0]))

                observation.append(curr_velocities[idx])
                observation.append(curr_directions[idx])


                observations_all.append(observation.copy())
                observation.clear()



    return observations_all

def get_spatio_temporal_frame_level_accuracies(model, frame_steps_temp, test_boxes, test_features, train_bbox_stats, train_velo_stats, bottom_border_only, file_list, save_dir, explain_anomalies):
    
    coco_names = np.load("./data/coco_names.npy")
    num_images = len(test_boxes)
    frame_scores_dict__all = {}

    for img_idx in tqdm.tqdm(range(frame_steps_temp, num_images)):
        
        filename = file_list[img_idx]

        curr_boxes, prev_boxes = test_boxes[img_idx], test_boxes[img_idx - frame_steps_temp]
        if curr_boxes is None:
            continue

        num_boxes_in_frame = len(curr_boxes)

        if num_boxes_in_frame == 0 or np.all(np.float32(curr_boxes.area().numpy().squeeze()) == 0.0):
            continue
        
        curr_features, prev_features = test_features[img_idx], test_features[img_idx - frame_steps_temp]
        prev_track_ids = [] if prev_features is None else [int(prev_feat[2]) for prev_feat in prev_features]

        object_scores, object_classes, object_confs, object_ids = [], [], [], []

        for idx in range(len(curr_boxes)):
            
            if np.float32(curr_boxes[idx].area().numpy().squeeze()) == 0.0:
                object_scores.append(-1.0)
                object_classes.append(None)
                object_confs.append(None)
                object_ids.append(None)
                continue
            
            curr_box, curr_feature = curr_boxes[idx], curr_features[idx]
            class_int = curr_feature[0]
            class_name = coco_names[class_int]
            conf = np.float16(curr_feature[1])
            curr_track_id = int(curr_feature[2])

            if curr_track_id in prev_track_ids:
                prev_idx = prev_track_ids.index(curr_track_id)
                prev_class_int = prev_features[prev_idx][0]

                if prev_class_int == class_int:
                    prev_box = prev_boxes[prev_idx]
                else: # ID switching issue in which new class label is assigned to the track....
                    prev_box = None

            else:
                prev_box = None

            # tic = time.time()
            score = compute_spatio_temporal_object_anomaly_score(model, frame_steps_temp, prev_box, curr_box, curr_feature, train_bbox_stats, train_velo_stats, bottom_border_only, filename, explain_anomalies)
            # toc = time.time()
            # object_time = (toc - tic)
            # print(f"\n\tComputing object score took {object_time:0.4f} seconds.")


            object_scores.append(score)
            object_classes.append(class_name)
            object_confs.append(conf)
            object_ids.append(curr_track_id)


        frame_scores_dict = {}

        if len(object_scores) == 0:
            frame_scores_dict[filename] = [0, None, None, None, None, None]

        else:

            for i in range(len(object_scores)):
                if object_scores[i] == -1.0:
                    continue

                score = object_scores[i]
                obj_class = object_classes[i]
                obj_conf = object_confs[i]
                obj_id = object_ids[i]
                bbox = curr_boxes.tensor[i].numpy()

                prev_results = frame_scores_dict.get(filename)
                if prev_results is None:
                    frame_scores_dict[filename] = [1, [score], [obj_class], [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]], [obj_conf], [obj_id]]
                else:
                    prev_results[0] += 1
                    prev_results[1].append(score)
                    prev_results[2].append(obj_class)
                    prev_results[3].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    prev_results[4].append(obj_conf)
                    prev_results[5].append(obj_id)
                    frame_scores_dict.update({ filename: prev_results })
        
        
        frame_scores_dict__all.update(frame_scores_dict)

        with open(save_dir, 'a') as f:
            for item in frame_scores_dict.items():
                f.write(str(item) + '\n')
        
    

    return frame_scores_dict__all

def compute_spatio_temporal_object_anomaly_score(model, frame_steps_temp, prev_box, curr_box, curr_feature, train_bbox_stats, train_velo_stats, bottom_border_only, filename, explain_anomalies):

    coco_names = np.load("./data/coco_names.npy")
    class_int = curr_feature[0]
    class_name = coco_names[class_int] 

    # Spatial Random Variables
    class_stats = train_bbox_stats.get(class_int)
    if class_stats is None: # Detected object class has not been encountered during training
        return 0.0
    
    bbox_area_stats = class_stats.get('area_stats')
    bbox_area_mean, bbox_area_std = bbox_area_stats.get('mean'), bbox_area_stats.get('std')

    box_size = get_bbox_area_cat(curr_box, bbox_area_mean, bbox_area_std)
    box_aspect_ratio = get_bbox_aspect_ratio_cat(curr_box)

    
    cell_ids, cell_boxes = locate_border(curr_box, bottom_border_only)
    assert len(cell_ids) > 0

    anomaly_infer = VariableElimination(model)
    object_score = 0.0
    
    for i in range(len(cell_ids)):
        cell = int(cell_ids[i])
        grid_cell = cell_boxes[i]
        intersect_cat = get_bbox_cell_intersection_cat(curr_box, grid_cell[0])

        # Temporal Random Variables
        if prev_box is None:
            box_velocity_cat, box_velocity_direction_cat = 'undefined', 'undefined'
        else:
            c_curr, c_prev = curr_box.get_centers().numpy().squeeze(), prev_box.get_centers().numpy().squeeze()
            box_velocity = calc_bbox_velocity__displacement(c_prev, c_curr, frame_steps_temp)

            class_velocity_stats = train_velo_stats.get(class_int)

            if class_velocity_stats is None:
                box_velocity_cat, box_velocity_direction_cat = 'undefined', 'undefined'
            else:
                box_velocity_cat = get_velocity_cat(box_velocity, class_velocity_stats[0], class_velocity_stats[1])
                box_velocity_direction_cat = 'none' if (box_velocity_cat == 'idle') else get_bbox_moving_direction_cat(c_prev, c_curr)
                    

        evidence_dict = {
            "Scene": 0, # Can be altered to account for multiple scenes
            "GridCell": cell,
            "BoxSize": box_size,
            "BoxAspectRatio": box_aspect_ratio,
            "Intersection": intersect_cat,

            "BoxVelocity": box_velocity_cat,
            "BoxVelocityDirection": box_velocity_direction_cat
        }



        class_prob = None
        # print("\n\nPredicted: ", class_name)
        # print("Conditional probability table at cell: ", str(covered_cells[i]))
        # print("")

        try:
            # tic = time.time()
            cpdt = anomaly_infer.query(variables=["Class"], evidence=evidence_dict, show_progress=False)
            # toc = time.time()
            # print(f"\n\t\tRaw inference took {inference_time:0.4f} seconds.")


            object_scores = cpdt.values
            # variables = cpdt.variables
            var_states = cpdt.state_names
            # print(cpdt)

        except:
            # print("\n")
            # print("Issue in image " + str(img_idx) + ":")
            # print("One or more categories not known as evidence:")
            # print(evidence_dict)
            # print("Recording probability score 0.0 for class " + class_name + " in cell " + str(cell) + ".")
            # print("\n")

            class_prob = 0.0
            
        
        if class_prob is None: # Inference query fro previous try block executed successfully
            try:
                class_prob = object_scores[var_states['Class'].index(class_name)]

            except Exception:
                # print("\n")
                # print("Issue in image " + str(img_idx) + ":")
                # print("Category '" + class_name + "' not found in probability table.")
                # print("Recording probability score 0.0 for class " + class_name + " in cell " + str(cell) + ".")
                # print("\n")

                class_prob = 0.0

        
        assert class_prob is not None

        if np.isnan(class_prob):
            class_prob = 0.0
        
        object_score += class_prob

        if explain_anomalies:
            explain_anomaly_score(anomaly_infer, evidence_dict, class_int, filename, class_prob)
    

    object_score /= (i + 1)
    
    assert object_score <= 1.0 # Probabilities must not be greater than 1


    return object_score






def explain_anomaly_score(anomaly_infer, evidence_dict, class_int, filename, class_prob):
    coco_names = np.load("./data/coco_names.npy")

    class_name = coco_names[class_int]
    cell = evidence_dict.get("GridCell")
    
    variables = ['BoxSize', 'BoxAspectRatio', 'Intersection', 'BoxVelocity', 'BoxVelocityDirection']
    CPTs = dict().fromkeys(variables)
    
    print("\n\nImage {0}: Class '{1}' detected in cell {2} with probability {3:.4f}.".format(filename, class_name, cell, class_prob), flush=True)
    print("Observed evidence: ", evidence_dict, flush=True)
    print("", flush=True)
    print("Breaking down impact  of individual variables:")

    variables.insert(0, 'Class')

    y_vals = np.zeros((len(variables), 1), np.float16)
    error = np.zeros_like(y_vals, np.float16)
    y_var_vals = []

    CPTs = dict().fromkeys(variables)
    evidence_dict.update({ 'Class': class_name })
    
    for variable in variables:
        try:
            var_val = evidence_dict.pop(variable)
            cpdt = anomaly_infer.query(     
                                        variables=[variable],
                                        evidence=evidence_dict,
                                        show_progress=False
                                    )

            evidence_dict.update({ variable: var_val })

            CPTs.update({ variable: [ cpdt.state_names[variable], cpdt.values ] })

            p_var = cpdt.values[cpdt.state_names[variable].index(var_val)]
            y_vals[variables.index(variable)] = p_var
            print("\n\t{0}({1}):\t{2}".format(variable, var_val, np.round(p_var, 4)), end=" ", flush=True)

            var_val_max = cpdt.state_names[variable][np.argmax(cpdt.values)]
            if p_var < np.max(cpdt.values):
                print("\t('{0}' more likely than '{1}')".format(var_val_max, var_val), end=" ", flush=True)
            
            error[variables.index(variable)] = np.max(cpdt.values)
            y_var_vals.append(var_val_max)
        
        except:
            print("\n\tNo prediction possible for variable {0} with value {1}. Assigning probability 0.0.".format(variable, var_val), end=" ", flush=True)
    


    return variables, y_vals, y_var_vals, error, CPTs
