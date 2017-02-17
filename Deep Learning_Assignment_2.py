# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:07:16 2017

@author: vsnalam
"""

#https://github.com/highspirits/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb
#----------------------------------------------------------------------------------------
#Previously in 1_notmnist.ipynb, we created a pickle with formatted datasets for training, 
#development and testing on the notMNIST dataset. The goal of this assignment is to progressively
#train deeper and more accurate models using TensorFlow.
#----------------------------------------------------------------------------------------
#for the command "import tensorflow as tf" to function, we need to first import tensorflow. 
#Below are the steps to import tensorflow into conda environment

#Step-1
#Open DOS prompt and point it to "C:\Sreedhar\Software\Anaconda Python"
#cd..
#cd..
#cd Sreedhar\Software\Anaconda Python

#Create an environment called tensorflow. The below command will create a folder in 
#"C:\Sreedhar\Software\Anaconda Python\Software\envs"
#create tensorflow

#Activate the new environment named "tensorflow"
#activate tensorflow

#Install Tensorflow package in this environment from the conda-forge channel
#https://github.com/conda-forge/tensorflow-feedstock
#conda config --add channels conda-forge

#Import Tensorflow into this tensorflow environment
#import tensorflow as tf

#Testing to see if tensorflow is installed correctly
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#----------------------------------------------------------------------------------------
#The above commands create a folder called tensorflow in Anaconda Python\Software\envs folder
#The folder contains all the files required for tensorflow library. 
#Now copy this folder to the current working directory
#C:\Sreedhar\Python\Code\Deep Learning

#After copying the tensorflow folder, the below command will execute without errors
#import tensorflow as tf
#----------------------------------------------------------------------------------------
import os
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#----------------------------------------------------------------------------------------
#First reload the data we generated in Deep learning_Assignment_1.py
#----------------------------------------------------------------------------------------
os.chdir("C:\Sreedhar\Python\Code\Deep Learning")

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels  = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels  = save['valid_labels']
    test_dataset  = save['test_dataset']
    test_labels   = save['test_labels']
    del save  # hint to help gc free up memory
    print('training set', train_dataset.shape, train_labels.shape)
    print('validation set', valid_dataset.shape, valid_labels.shape)    
    print('test set', test_dataset.shape, test_labels.shape)
#----------------------------------------------------------------------------------------
# 1 hot encoding process
#----------------------------------------------------------------------------------------
#a = train_labels[0:10]
#a 
#array([ 4.,  9.,  6.,  2.,  7.,  3.,  5.,  9.,  6.,  4.], dtype=float32)

#a[:, None]
#array([[ 4.],
#       [ 9.],
#       [ 6.],
#       [ 2.],
#       [ 7.],
#       [ 3.],
#       [ 5.],
#       [ 9.],
#       [ 6.],
#       [ 4.]], dtype=float32)

#None is equivalent to newaxis
#for validation, (np.newaxis == None) returns True

#np.arange(10) == a[:, None]
#array([[False, False, False, False,  True, False, False, False, False,
#        False],
#       [False, False, False, False, False, False, False, False, False,
#         True],
#       [False, False, False, False, False, False,  True, False, False,
#        False],
#       [False, False,  True, False, False, False, False, False, False,
#        False],
#       [False, False, False, False, False, False, False,  True, False,
#        False],
#       [False, False, False,  True, False, False, False, False, False,
#        False],
#       [False, False, False, False, False,  True, False, False, False,
#        False],
#       [False, False, False, False, False, False, False, False, False,
#         True],
#       [False, False, False, False, False, False,  True, False, False,
#        False],
#       [False, False, False, False,  True, False, False, False, False,
#        False]], dtype=bool)

#(np.arange(10) == a[:, None]).astype(np.float32)
#array([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
#       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
#       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
#       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)
#----------------------------------------------------------------------------------------
#Reformat into a shape that's more adapted to the models we're going to train:
#   data as a flat matrix,
#   labels as float 1-hot encodings.
#----------------------------------------------------------------------------------------
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    #changing the size from 3D array to 2D array
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    
    #1 hot encoding    
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
    
train_dataset, train_labels = reformat(train_dataset, train_labels)   
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset,  test_labels  = reformat(test_dataset,  test_labels)

print ("Training Dataset", train_dataset.shape, train_labels.shape)
#Training Dataset (200000, 784) (200000, 10)
print ("Validation Dataset", valid_dataset.shape, valid_labels.shape)
#Validation Dataset (10000, 784) (10000, 10)
print ("Test Dataset", test_dataset.shape, test_labels.shape)
#Test Dataset (10000, 784) (10000, 10)












