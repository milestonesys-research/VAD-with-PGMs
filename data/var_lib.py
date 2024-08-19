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

def init():
    ### GLOBAL VARIABLES

    global METADATA_DIR__ROOT
    METADATA_DIR__ROOT = ""

    global TRAIN_DIR_ROOT, TEST_DIR_ROOT
    TRAIN_DIR_ROOT, TEST_DIR_ROOT = "", ""

    global IMG_WIDTH, IMG_HEIGHT
    IMG_WIDTH, IMG_HEIGHT = 0, 0
    
    global GRID_CELL_LOOKUP
    GRID_CELL_LOOKUP = 0

    global LOAD_CSV_FILE_DIR, SAVE_CSV_FILE_DIR
    LOAD_CSV_FILE_DIR, SAVE_CSV_FILE_DIR = "", ""

    global SAVE_RESULTS_DIR
    SAVE_RESULTS_DIR = ""



### CATALOGUES

DATASET_CATALOGUE = [
    'ToySet'
    #, 'name'
]

METADATA_DIR_CATALOGUE = [
    './data/meta_data/{0}/'
    #, 'path/to/bounding/boxes/and/track/data/'
]

TRAIN_DIR_CATALOGUE = [
    './data/images/{0}/',
    #, 'path/to/training/images/'
]

TEST_DIR_CATALOGUE = [
    './data/images/{0}/',
    #, 'path/to/test/images/'
]

DIMENSION_CATALOGUE =   [
    'spatial',
    'spatio-temporal'
]

MODEL_VARIANT_CATALOGUE =  [

    # SPATIAL MODELS
    's01',          # 00 #  Spatial Model from Ablation Study section in paper
    # SPATIO-TEMPORAL MODELS
    'st01',         # 01 #  Spatio-Temporal/main model from paper
                                    
]

