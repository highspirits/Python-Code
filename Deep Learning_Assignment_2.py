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
#----------------------------------------------------------------------------------------
#The above commands create a folder called tensorflow in Anaconda Python\Software\envs folder
#The folder contains all the files required for tensorflow library. 
#Now copy this folder to the current working directory
#C:\Sreedhar\Python\Code\Deep Learning

#After copying the tensorflow folder, the below command will execute without errors
#import tensorflow as tf

#The above folder copying option did not work as the import command executed successfully 
#but tf.constant command did not work. 

#So I first pointed the current working directory to the environment folder having tensorflow
#installed and then imported tensorflow. Below are the commands for this - 

#import os
#os.chdir("C:\Sreedhar\Software\Anaconda Python\Software\envs")
#import tensorflow as tf

#Then the tensorflow commands are executed successfully 

#After this, change the working directory to required other folder
#----------------------------------------------------------------------------------------
from __future__ import print_function
import os
os.chdir("C:\Sreedhar\Software\Anaconda Python\Software\envs")
import tensorflow as tf
os.chdir("C:\Sreedhar\Python\Code\Deep Learning")
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range

#----------------------------------------------------------------------------------------
#First reload the data we generated in Deep learning_Assignment_1.py
#----------------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------------
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
#---------------------------------------------------------------------------------------
train_subset= 10000

graph = tf.Graph()
with graph.as_default():
    #Input data
    #Load training, validation and testing data into constants that are attached to 
    #the graph. 
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :]) #size = (10000 X 784)
    tf_train_labels  = tf.constant(train_labels[:train_subset])     #size = (10000 X 10)
    tf_valid_dataset = tf.constant(valid_dataset)                   #size = (10000 X 784)
    tf_valid_labels  = tf.constant(valid_labels)                    #size = (10000 X 10)
    tf_test_dataset  = tf.constant(test_dataset)                    #size = (10000 X 784)
    
    print('Dataset shapes after subsetting')    
    print ("Training Dataset", tf_train_dataset.shape, tf_train_labels.shape)
    #Training Dataset (100000, 784) (100000, 10)
    print ("Validation Dataset", tf_valid_dataset.shape, tf_valid_labels.shape)
    #Validation Dataset (10000, 784) (10000, 10)
    print ("Test Dataset", tf_test_dataset.shape)
    #       (10000 X 10)    
    
    #Variables - Weights and biases
    #These are the parameters that we will train. 
    #The Weight matrix will be initialized using random normal distribution. 
    #The biases will be initialized to zeros
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels])) 
                                                #size= (784 X 10)
    biases  = tf.Variable(tf.zeros([num_labels]))                   
                            #size = (1X10)
    
    #with tf.Session() as session:
    #    print(session.run(tf.shape(weights)))
    
    #Training Computation
    #We will multiply input with weights and add biases - (WX+b). 
    #We then compute the softmax and cross-entropy. 
    #We then take the average of this cross-entropy across all training examples. This is loss
    logits = tf.matmul(tf_train_dataset, weights) + biases
    #                  ( 10000 X 784) X (784 * 10)  + (1 X 10)
    #                   (10000 X 10) + (1 X 10)
    #                   (10000 X 10)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))
    
    #Optmizer
    #We will compute the min of this loss using gradient descent
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset * weights) + biases)
    #                                            (10000 X 784) * (784 X 10) + (1 X 10)
    #                                            (10000 X 10) + (1 X 10)
    #                                            (10000 X 10)
    test_prediction  = tf.nn.softmax(tf.matmul(tf_test_dataset * weights) + biases)   
    #                                           (10000 X 784) * (784 X 10) + (1 X 10)
    #                                           (10000 X 10)
#----------------------------------------------------------------------------------------
#lets run this computation and iterate:
#Running Gradient descent algorithm
#number of steps required for grandient to identify the local minima is less compared to 
#Stochastic Gradient Descent (see later steps). But the process runs on entire input dataset
#whcih is time cosuming and process intensive. 
#---------------------------------------------------------------------------------------
num_steps = 801

def accuracy(predictions, labels):
    #formula for accuracy = ((correct predictions)/total predictions) * 100
    #argmax returns the position of max value in a table. argmax(x, 1) gives the max value row wise
    # a = ([[0, 1, 2],
    #      [3, 4, 5]]) --> argmax (a, 1) = [2, 2]
    #softmax function returns the probibility. The position of the max probability is compared with the 
    #position of the value '1' in labels. If they match, prediction is accurate, else inaccurate. 
    score = (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))/(predictions.shape[0]) * 100
    return score
    
with tf.Session(graph = graph) as session:
    #Initialize the variables in graph: Weights and biases
    tf.global_variables_initializer().run()
    
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, losss, predictions = session.run([optimizer, loss, train_prediction])
        if (step%100 == 0):
            print ('Loss at step %d: %f' %(step, losss))
            print('Training Accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
            
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            
            print('Validation Accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            print('Testing Accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
            
#Loss at step 0: 15.580845
#Training Accuracy: 11.1%
#Loss at step 100: 2.297943
#Training Accuracy: 71.4%
#Loss at step 200: 1.860387
#Training Accuracy: 74.8%
#Loss at step 300: 1.627600
#Training Accuracy: 76.4%
#Loss at step 400: 1.468904
#Training Accuracy: 77.1%
#Loss at step 500: 1.348289
#Training Accuracy: 77.8%
#Loss at step 600: 1.252118
#Training Accuracy: 78.3%
#Loss at step 700: 1.173102
#Training Accuracy: 78.9%
#Loss at step 800: 1.106789
#Training Accuracy: 79.2%
#-----------------------------------------------------------------------------------------------

            
    
    
    
    
    