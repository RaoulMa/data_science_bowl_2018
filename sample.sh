#!/bin/bash
# Author: Raoul Malm
# Description: Bash script to train the neural network for
# image segmentation of the 2018 Data Science Bowl.

# pretrained models
#nn_name = 'nn0_256_256_3'
#nn_name = 'nn0_384_384_3'
#nn_name = 'nn0_512_512_3'
#nn_name = 'tmp' # for testing purpose

# Training and prediction for testing purposes on a reduced data set
python main.py
    --name tmp \
#    --train_imgs 20 \
#    --test_imgs 20 \
#    --train \
    --predict \
    --epoch 1.0

# Training and prediction on the full data set
#python main.py \
#--name nn0_384_384_3 \
#--predict \
#--epoch 10.0

