# syntax=docker/dockerfile:1
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \ 
    apt install -y python3 python3-pip python-is-python3 && \
    apt install -y ffmpeg libsm6 libxext6 && \
    apt-get -qq clean

# Installing torch and torchvision
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Installing detectron2
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Installing OpenCV
RUN pip3 install opencv-python

# Installing pgmpy
RUN pip3 install pgmpy

# Environment variables
ENV WORK=/workdir

# Set workdir to home
WORKDIR ${WORK}

CMD ["bash", "--login"]
