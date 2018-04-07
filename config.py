#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raoul Malm
Description: Configuration file to set all global variables to default values. 
"""

import os    # For filepath, directory handling

# Global constants.
IMG_WIDTH = 384       # Default image width
IMG_HEIGHT = 384      # Default image height
IMG_CHANNELS = 3      # Default number of channels
CW_DIR = os.getcwd()  
#TRAIN_DIR = os.path.join(os.path.dirname(CW_DIR), 'input', 'stage1_train')
#TEST_DIR = os.path.join(os.path.dirname(CW_DIR), 'input', 'stage1_test')
TRAIN_DIR = os.path.join(CW_DIR, 'data', 'stage1_train')
TEST_DIR = os.path.join(CW_DIR, 'data', 'stage1_test')
IMG_TYPE = '.png'         # Image type
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks
LOGS_DIR_NAME = 'logs'    # Folder name for TensorBoard summaries 
SAVES_DIR_NAME = 'saves'  # Folder name for storing network parameters
SEED = 123                # Random seed for splitting train/validation sets
MB_SIZE = 16              # Mini Batch Size
    
# Global variables.
min_object_size = 1       # Minimal nucleous size in pixels
nn_name = 'tmp'           # Default neural network name
n_epoch = 1.0               # Default number of epochs for training
#x_train = []
#y_train = []
#x_test = []
#y_test_pred_proba = {}
#y_test_pred = {}
