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

from core.core import *
from core.network import construct_network_model_bayesian
from data.dataloader import load_objects, get_confidence_score_threshold

import data.var_lib as var_lib
import os


def main(args):

    
    #########################################
    ### LOAD TRAINING DATASET
    #########################################


    print("\n\nProcessing training data ...", flush=True)
    image_name_list = np.loadtxt(var_lib.TRAIN_DIR_ROOT.format(args['dataset']) + "toy00__frame_names.txt", dtype=str)
    image_name_list.sort()

    
    if args['dynamic_conf_thresh']:
        print("--> Calculating confidence scores of training data ...", flush=True)
        conf_thresh_values = get_confidence_score_threshold(args['dataset'])
    else:
        conf_thresh_values = (0.5, 0.5)
    

    print("--> Loading training meta-data ...", flush=True)
    train_boxes_list, train_features_list, train_boxes_count = load_objects(args['dataset'], 'train', image_name_list, conf_thresh_values)

    print("\nProcessed {0} training frames and {1} detections.".format(len(image_name_list), train_boxes_count), flush=True)

    print("\n\n")
    print("Calculating global statistics of training data ...", flush=True)
    print("--> Spatial statistics ...", flush=True)

    if args['dimension'] == 'spatial':
        train_bbox_stats = get_bbox_stats(train_boxes_list[::args['frame_steps_train']], train_features_list[::args['frame_steps_train']])

    else:
        train_bbox_stats = get_bbox_stats(train_boxes_list[args['frame_steps_temp']::args['frame_steps_train']], train_features_list[args['frame_steps_temp']::args['frame_steps_train']])
        print("--> Temporal statistics ...", flush=True)
        train_velo_stats = get_velocity_stats(args['frame_steps_temp'], train_boxes_list[args['frame_steps_temp']::args['frame_steps_train']], train_features_list[args['frame_steps_temp']::args['frame_steps_train']])

    
    print("\n\n")

    #########################################
    ### LOAD BAYESIAN NETWORK
    #########################################

    if bool(args['load_from_csv']) == True:
        bayesian_model, num_observations = construct_network_model_bayesian(args['model'], None, True, False)

    else:

        if args['dimension'] == 'spatial':
            observations = generate_spatial_observations(train_bbox_stats, train_boxes_list[::args['frame_steps_train']], train_features_list[::args['frame_steps_train']], args['bottom_border_only'])
        else:
            observations = generate_spatio_temporal_observations(args['frame_steps_temp'], train_bbox_stats, train_velo_stats, train_boxes_list[args['frame_steps_temp']::args['frame_steps_train']], train_features_list[args['frame_steps_temp']::args['frame_steps_train']], args['bottom_border_only'])
                
        bayesian_model, num_observations = construct_network_model_bayesian(args['model'], observations, bool(args['load_from_csv']), bool(args['save_to_csv']))


    print("\n")
    print("Number of observations:", num_observations)
    print("Model okey? ", bayesian_model.check_model())



    #########################################
    ### LOAD TEST DATASET
    #########################################

    image_name_list = []
    image_name_list = np.loadtxt(var_lib.TEST_DIR_ROOT.format(args['dataset']) + "toy00__frame_names.txt", dtype=str)
    image_name_list.sort()

    num_images = len(image_name_list)


    print("\n\n\n")
    print("Processing test data ...", flush=True)
    print("--> Loading test meta-data ...", flush=True)
    test_boxes_list, test_features_list, test_boxes_count = load_objects(args['dataset'], 'test', image_name_list, conf_thresh_values)
    
    print("\n")
    print("Test dataset loaded:")
    print("- Total number of images: ", num_images)
    print("- Total number of objects: ", test_boxes_count)


    print("\n\n")
    print("Starting Inference ...", flush=True)
    print("--> Calculating object probability scores for all test objects ...", flush=True)


    save_dir = var_lib.SAVE_RESULTS_DIR.format(args['model'])

    # Delete file if it already exists
    if os.path.isfile(save_dir):
        os.remove(save_dir)

    if args['dimension'] == 'spatial':
        _ = get_spatial_frame_level_accuracies(bayesian_model, test_boxes_list, test_features_list, train_bbox_stats, args['bottom_border_only'], image_name_list, save_dir, args['explain_anomalies'])
    else:
        _ = get_spatio_temporal_frame_level_accuracies(bayesian_model, args['frame_steps_temp'], test_boxes_list, test_features_list, train_bbox_stats, train_velo_stats, args['bottom_border_only'], image_name_list, save_dir, args['explain_anomalies'])



    print("\n\nNetwork structure summary:")
    print(bayesian_model.nodes())
    print(bayesian_model.edges())
    print("\nResults saved in root directory:")
    print(var_lib.SAVE_RESULTS_DIR)
    print("\n\n")


    return 0


    