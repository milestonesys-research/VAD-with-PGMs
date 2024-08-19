#!/bin/sh
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
set -e

# 
# usage: parse_script.py [-h] ...
# 
# optional arguments:
#   -h, --help            show this help message and exit



python parse_script.py \
--dataset 0 --img-width 1280 --img-height 720 --csize 80 \
--model 1 --dimension 1 --dynamic-conf-thresh 1 --bottom-border-only 1 \
--frame-steps-train 1 --frame-steps-temp 1 \
--explain-anomalies 0

# --load-from-csv 0
# --save-to-csv 0



echo 'The End.'
