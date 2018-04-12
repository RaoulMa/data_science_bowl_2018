#!/bin/bash
# Author: Raoul Malm
# Description: Bash script to train the neural network for
# image segmentation of the 2018 Data Science Bowl.

# pretrained available models:
#--name nn0_256_256_3
#--name nn0_384_384_3
#--name nn0_512_512_3
#--name tmp # for testing purpose

# options of running main.py:
#--name: argument = name of the model
#--load: load a pretrained model
#--train: train the model
#--predict: make predictions
#--epoch: argument = number of epochs for training

# Training and prediction for testing purposes on a reduced data set
python main.py --name tmp \
    --train_imgs 20 \
    --test_imgs 20 \
    --train \
    --predict \
    --epoch 1.0

# Training and prediction on the full data
#python main.py --name tmp \
#--load
#--train
#--predict \
#--epoch 1.0

# Continue training and prediction on the full data set where the model is loaded from a file
#python main.py --name nn0_384_384_3 \
#--load
#--train
#--predict \
#--epoch 1.0

