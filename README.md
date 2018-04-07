# Data Science Bowl 2018
This code implements a U-Net shaped convolutional neural network in TensorFlow for 
nuclei segmentation of the images provided by the 2018 Data Science Bowl, see
https://www.kaggle.com/c/data-science-bowl-2018/data. 

## Abstract
The 2018 Data Science Bowl "Find the nuclei in divergent images to advance medical discovery" provides in its first stage a training and test data set consisting of 670 and 65 microscopic images of varying size showing ensembles of cells and their nuclei. For the training images the nuclei are segmented by humans such that we know their number and location within each image. The goal is to find the correct number and location of all nuclei shown in the test images. The performance of an algorithm is evaluated on the mean average precision at different intersection over union (IoU) thresholds, which will be referred to as the score in the following.

For the task we implement a deep neural network of the U-Net type consisting of several convolutional and max-pooling layers. The network is written in TensorFlow. Each model can be saved/loaded and the training process can be visualized with TensorBoard. The input of the network are images of shape (height, width, channels) while the output are corresponding binary masks of shape (height, width, 1).

Prior to training the network we resize, normalize and transform the images. We use 10% of the training data for validation. Furthermore, we implement data augmentation by making use of translations, rotations, horizontal/vertical flipping and zoom. Choosing an image size of 384x384 pixels the network requires roughly 30 training epochs before the training seems to converge. The network achieves a score of 0.56/0.352 on the validation/test set. A major reason for the score discrepancy can be explained by overlapping/touching nuclei that are identified as a single nucleous by the current implementation. Furthermore, we have not tuned the hyperparameters, so there is still a lot of room for improvement.

## Run the code

The code is tested with python 3.6.5, Tensorflow 1.4.1 and Tensorboard 1.7.0. Models are saved in the folder 'saves', logs for Tensorboard are saved in the folder 'logs' and the image data is contained in the folder 'data'. To execute the code the following files are relevant:

- sample.sh is a bash script, which trains a model called 'tmp' and makes predictions including the creation of the submission file. You can modify the file for your purposes. 

- analysis.ipynb is a jupyter notebook which can also be used for training the model. In addition it includes code for the analysis and visualization of the problem. 

- config.py contains global variables like the file paths to the image data. You can modify the file for your purposes. 





