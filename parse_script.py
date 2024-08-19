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

import os, argparse
import numpy as np

import data.var_lib as var_lib

from main_script import main



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dataset', type=int, default=0, help='Dataset Name: StreetScene (0)')

    parser.add_argument('--img-width', type=int, default=720, help='Image Width: StreetScene (1280)')
    parser.add_argument('--img-height', type=int, default=480, help='Image Height: StreetScene (720)')
    parser.add_argument('--csize', type=int, default=40, help='Cell Size: StreetScene (40, 80)')

    parser.add_argument('--dynamic-conf-thresh', type=int, default=0, help='Flag indicating whether to dynamically set the confidence threshold to filter out potential FPs.')
    parser.add_argument('--dimension', type=int, default=0, help='Spatial (0), Spatio-Temporal (1)')
    parser.add_argument('--model', type=int, default=0, help='Consult var_lib.py.')
        
    parser.add_argument('--bottom-border-only', type=int, default=1, help='Consider only cells that intersect with the bottom edge of a box.')
    parser.add_argument('--frame-steps-train', type=int, default=1, help='Slice factor limiting amount of training frames to use.')
    parser.add_argument('--frame-steps-temp', type=int, default=1, help='Time/frame step. For spatio-temporal setting only.')

    parser.add_argument('--load-from-csv', type=bool, default=False, help='Load observations from file.')
    parser.add_argument('--save-to-csv', type=bool, default=False, help='Save observations to file.')

    parser.add_argument('--explain-anomalies', type=int, default=0, help='Set to 1 if explanation of anomaly scores is needed. Note: This will clutter your terminal window because the messages are printed to stdout.')


    args = parser.parse_args()
    
    return args




if __name__ == "__main__":
    args = get_args_parser()
    dict_vars = vars(args)

    ds_id = args.dataset
    dataset_name = var_lib.DATASET_CATALOGUE[ds_id]
    dict_vars.update({'dataset': dataset_name})

    dynamic_conf_thresh = args.dynamic_conf_thresh
    dict_vars.update({'dynamic_conf_thresh': True if (dynamic_conf_thresh == 1) else False})

    dimension_name = var_lib.DIMENSION_CATALOGUE[args.dimension]
    dict_vars.update({'dimension': dimension_name})

    model_name = var_lib.MODEL_VARIANT_CATALOGUE[args.model]
    dict_vars.update({'model': model_name})

    bottom_border = args.bottom_border_only
    dict_vars.update({'bottom_border_only': True if (bottom_border == 1) else False})

    explain_anomalies = args.explain_anomalies
    dict_vars.update({'explain_anomalies': True if (explain_anomalies == 1) else False})


    # INITIALIZE GLOBAL VARIABLES
    var_lib.init()

    var_lib.METADATA_DIR__ROOT = os.path.abspath(var_lib.METADATA_DIR_CATALOGUE[ds_id]) + '/'
    var_lib.TRAIN_DIR_ROOT, var_lib.TEST_DIR_ROOT = os.path.abspath(var_lib.TRAIN_DIR_CATALOGUE[ds_id]) + '/', os.path.abspath(var_lib.TEST_DIR_CATALOGUE[ds_id]) + '/'
    var_lib.IMG_WIDTH, var_lib.IMG_HEIGHT = args.img_width, args.img_height
    dict_vars.pop('img_width')
    dict_vars.pop('img_height')
    
    var_lib.GRID_CELL_LOOKUP = np.zeros((var_lib.IMG_HEIGHT, var_lib.IMG_WIDTH), np.int32)
    cell = 1

    for row in range(args.csize, var_lib.IMG_HEIGHT + args.csize, args.csize):
        for col in range(args.csize, var_lib.IMG_WIDTH + args.csize, args.csize):
            var_lib.GRID_CELL_LOOKUP[(row - args.csize):(row), (col - args.csize):(col)] = cell
            cell += 1


    ##########################################################
    # Following path variables to be set by user
    ##########################################################
    var_lib.LOAD_CSV_FILE_DIR = "."                 # Optional
    var_lib.SAVE_CSV_FILE_DIR = "."                 # Optional
    var_lib.SAVE_RESULTS_DIR = "./output/results/"  # Required

    if not os.path.exists(var_lib.SAVE_RESULTS_DIR):
        os.makedirs(var_lib.SAVE_RESULTS_DIR)

    var_lib.SAVE_RESULTS_DIR += "results--object-scores--{}.txt"

    print("\nStarting task...")
    print("\n\n")


    main(dict_vars)


    print("\n\n")
    print("Parsed arguments:\n")

    for val in dict_vars.items():
        print("\t", val)


    print("\n\nDone with task.\n\n\n")
