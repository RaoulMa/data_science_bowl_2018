#!/bin/bash
# Author: Raoul Malm
# Description: Bash script to train the neural network for
# image segmentation of the 2018 Data Science Bowl.

# options of running main.py:
#--name: argument = name of the model
#--load: load a pretrained model
#--train: train the model
#--predict: make predictions
#--epochs: argument = number of epochs for training

# Training of 5 models with 10-fold cross validation for 50 epochs
python main.py --name nn0_256_256 nn1_256_256 nn2_256_256 nn3_256_256 nn4_256_256 \
--train \
--epochs 50.0

# Prediction of 5 trained models
#python main.py --name nn0_256_256 nn1_256_256 nn2_256_256 nn3_256_256 nn4_256_256 \
#--predict 

# Training and prediction for testing purposes on a reduced data set
#python main.py --name tmp \
#    --train_imgs 50 \
#    --test_imgs 50 \
#    --train \
#    --predict \
#    --epochs 2.0

# Training and prediction on the full data
#python main.py --name tmp \
#--load \
#--train \
#--predict \
#--epochs 1.0

# Continue training and prediction on the full data set where the model is loaded from a file
#python main.py --name nn0_384_384_3 \
#--load \
#--train \
#--predict \
#--epochs 1.0




