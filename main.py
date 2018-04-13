#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raoul Malm
Description: Main source file which implements the U-Net shaped 
convolutional network for image segmentation of the 2018 Data Science Bowl.
"""

# Import necessary modules and set global constants and variables. 
import argparse
import tensorflow as tf            
import pandas as pd                 
import numpy as np                                       
import sys
import sklearn.model_selection     # For using KFold
import datetime                    # To measure running time 
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling

# Import global variables stored in configuration file.
import config
from config import TRAIN_DIR, TEST_DIR, IMG_DIR_NAME, MASK_DIR_NAME, SEED, \
                   NN_NAME, N_EPOCHS, CV_NUM
                   
# Import useful methods.
from utils import read_train_data_properties, read_test_data_properties, \
                  load_raw_data, preprocess_raw_data, mask_to_rle, \
                  trsf_proba_to_binary
                  
# Import the neural network class.
from neuralnetwork import NeuralNetwork

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='U-Net for image segmentation.')

    parser.add_argument('-n', '--name', default=NN_NAME, nargs='+',
                        help='name of the neural network as an argument')
    parser.add_argument("-l", "--load", action="store_true",
                    help="load a model and continue training") 
    parser.add_argument("-t", "--train", action="store_true",
                    help="train the model")
    parser.add_argument("-p", "--predict", action="store_true",
                    help="make model prediction and create submission file")
    parser.add_argument("--train_imgs", type=int,
                    help="specify how many train images should be used")
    parser.add_argument("--test_imgs", type=int,
                    help="specify how many test images should be used")
    parser.add_argument("-e", "--epochs", type=float, default=N_EPOCHS,
                    help="specify how many test images should be used")
    
    args = parser.parse_args()
    nn_name = args.name      # neural network name
    n_epochs = args.epochs    # number of epochs
    
    # Display versions.
    print('python version: {}'.format(sys.version))
    print('tensorflow version: {}'.format(tf.__version__))
    print(args)
    
    # Basic properties of images/masks. 
    train_df = read_train_data_properties(TRAIN_DIR, IMG_DIR_NAME, MASK_DIR_NAME)
    test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)

    # Reduce the amount of data for testing purpose
    if args.train_imgs is not None:
        train_df = train_df[:args.train_imgs]
    if args.test_imgs is not None:
        test_df = test_df[:args.test_imgs]
    
    # Read images/masks from files and resize them. Each image and mask 
    # is stored as a 3-dim array where the number of channels is 3 and 1,
    # respectively.
    print('\nRead data. ')
    x_train, y_train, x_test = load_raw_data(train_df, test_df)
    
    # Normalize all images and masks. There is the possibility to transform images 
    # into the grayscale sepctrum and to invert images which have a very 
    # light background.
    x_train, y_train, x_test = preprocess_raw_data(x_train, y_train, x_test,
                                                   invert=True)
      
    if args.train:
        # Create and start training of a new neural network, or continue training of
        # a pretrained model.
        
        # Implement cross validations
        cv_num = CV_NUM
        kfold = sklearn.model_selection.KFold(cv_num, shuffle=True, 
                                              random_state=SEED)
        
        for i,(train_index, valid_index) in enumerate(kfold.split(x_train)):
        
            # Start timer
            start = datetime.datetime.now();
        
            # Split data into train and validation sets
            x_trn = x_train[train_index]
            y_trn = y_train[train_index]
            x_vld = x_train[valid_index]
            y_vld = y_train[valid_index]
            
            if i<len(nn_name): # Choose a certain fold.

                if not args.load:
                    # Create and start training of a new model.
                    print('\nStart training a new model.')
    
                    # Create instance of neural network.
                    u_net = NeuralNetwork(nn_name=nn_name[i], log_step=0.3)
                    u_net.build_graph() # Build graph.
        
                    # Start tensorflow session.
                    with tf.Session(graph=u_net.graph) as sess: 
                        u_net.attach_saver() # Attach saver tensor.
                        u_net.attach_summary(sess) # Attach summaries.
                        sess.run(tf.global_variables_initializer()) 
                        
                        if n_epochs > 1.0:
                            n_epoch_org = 1.0
                            n_epoch_aug = n_epochs-1.0
                        else:
                            n_epoch_org = n_epochs
                            n_epoch_aug = 0.0
                            
                        # Training on original data.
                        u_net.train_graph(sess, x_trn, y_trn, x_vld, y_vld, 
                                          n_epoch=n_epoch_org)
                        
                        # Save parameters, tensors, summaries.
                        u_net.save_model(sess)
                        
                        for _ in range(int(n_epoch_aug // 1.0)):
                            # Training on augmented data.
                            u_net.train_graph(sess, x_trn, y_trn, x_vld, 
                                              y_vld, n_epoch=1.0,
                                              train_on_augmented_data=True)
                            # Save parameters, tensors, summaries.
                            u_net.save_model(sess) 
        
                if args.load:
                    # Continue training of a pretrained model.
                    print('\nContinue training of a loaded model.')
                    
                    u_net = NeuralNetwork() 
                    sess = u_net.load_session_from_file(nn_name[i])
                    u_net.attach_saver() 
                    u_net.attach_summary(sess) 
  
                    for _ in range(int(n_epochs//1.0)):
                        # Training on augmented data.
                        u_net.train_graph(sess, x_trn, y_trn, x_vld, y_vld, 
                                          n_epoch=1.0,
                                          train_on_augmented_data = True)
                        
                        # Save parameters, tensors, summaries.
                        u_net.save_model(sess) 
        
        print('Total running time: '.format(datetime.datetime.now() - start))
        
    
    if args.predict:
        
        # Load neural network, make prediction for test masks, resize predicted
        # masks to original image size and apply run length encoding for the
        # submission file. 
        
        print('\nMake prediction and create submission file.')
    
        # Soft voting majority.
        for i,mn in enumerate(nn_name):
            u_net = NeuralNetwork()
            sess = u_net.load_session_from_file(mn)
            if i==0: 
                y_test_pred_proba = u_net.get_prediction(sess, x_test)/len(nn_name)
            else:
                y_test_pred_proba += u_net.get_prediction(sess, x_test)/len(nn_name)
            sess.close()
        
        y_test_pred = trsf_proba_to_binary(y_test_pred_proba)
        print('y_test_pred.shape = {}'.format(y_test_pred.shape))
        
        # Resize predicted masks to original image size.
        y_test_pred_original_size = []
        for i in range(len(y_test_pred)):
            res_mask = trsf_proba_to_binary(skimage.transform.resize(np.squeeze(y_test_pred[i]),
                (test_df.loc[i,'img_height'], test_df.loc[i,'img_width']), 
                mode='constant', preserve_range=True))
            y_test_pred_original_size.append(res_mask)
        y_test_pred_original_size = np.array(y_test_pred_original_size)
        
        print('y_test_pred_original_size.shape = {}'.format(y_test_pred_original_size.shape))
           
        # Run length encoding of predicted test masks.
        test_pred_rle = []
        test_pred_ids = []
        for n, id_ in enumerate(test_df['img_id']):
            config.min_object_size = 30*test_df.loc[n,'img_height']*test_df.loc[n,'img_width']/(256*256)
            rle = list(mask_to_rle(y_test_pred_original_size[n]))
            test_pred_rle.extend(rle)
            test_pred_ids.extend([id_]*len(rle))
        
        print('test_pred_ids.shape = {}'.format(np.array(test_pred_ids).shape))
        print('test_pred_rle.shape = {}'.format(np.array(test_pred_rle).shape))
            
            
        # Create submission file
        sub = pd.DataFrame()
        sub['ImageId'] = test_pred_ids
        sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('sub-dsbowl2018.csv', index=False)
        sub.head()
            
