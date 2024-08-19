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

import pandas as panda
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

import data.var_lib as var_lib



def construct_network_model_bayesian(network_variant, observations, load_from_csv=False, save_csv_file=False):
    
    # Spatial Model
    if network_variant == 's01':
        categories = ['Scene', 'GridCell', 'Class', 'BoxSize', 'BoxAspectRatio', 'Intersection']
        edges = [
                    ('Scene', 'GridCell'),
                    ('GridCell', 'Class'),
                    ('GridCell', 'Intersection'),
                    ('Class', 'BoxSize'),
                    ('Class', 'BoxAspectRatio'),
                    ('Class', 'Intersection')
                ]

    # Spatio-Temporal Models
    elif network_variant == 'st01':
        categories = ['Scene', 'GridCell', 'Class', 'BoxSize', 'BoxAspectRatio', 'Intersection', 'BoxVelocity', 'BoxVelocityDirection']
        edges = [
                    ('Scene', 'GridCell'),
                    
                    ('GridCell', 'Class'),
                    ('GridCell', 'Intersection'),
                    ('GridCell', 'BoxSize'),
                    
                    ('Class', 'Intersection'),
                    ('Class', 'BoxSize'),
                    ('Class', 'BoxAspectRatio'),

                    ('Class', 'BoxVelocity'),
                    ('GridCell', 'BoxVelocity'),
                    ('GridCell', 'BoxVelocityDirection')
                ]
    


    if load_from_csv:
        data_frame = panda.read_csv(var_lib.LOAD_CSV_FILE_DIR)

    else:
        data_frame = panda.DataFrame(observations, columns=categories)

        if save_csv_file:
            data_frame.to_csv(var_lib.SAVE_CSV_FILE_DIR, columns=categories)


    bayesian_model = BayesianNetwork()
    bayesian_model.add_nodes_from(categories)
    bayesian_model.add_edges_from(edges)

    bayesian_model.fit(data_frame, estimator=MaximumLikelihoodEstimator)



    return bayesian_model, len(data_frame)

