#!/bin/sh

echo "######################################"
echo "Running installation script ..."
echo "######################################"

echo ""
echo "### (1/4) Installing torch and torchvision ..."
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html



echo ""
echo "### (2/4) Installing detectron2 ..."
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html



echo ""
echo "### (3/4) Installing OpenCV ..."
pip install opencv-python



echo ""
echo "### (4/4) Installing pgmpy ..."
pip install pgmpy

echo ""
echo "######################################"
echo "Done."
echo "######################################"