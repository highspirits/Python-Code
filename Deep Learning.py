# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:10:51 2016

@author: vsnalam
"""
#-------------------------------------------------------------------------------------------
#Udacity Deep Learning
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb
#-------------------------------------------------------------------------------------------

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import metrics

os.chdir("C:\Sreedhar\Python\Code\Deep Learning")

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline
#-------------------------------------------------------------------------------------------
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

#-------------------------------------------------------------------------------------------
#Extract the dataset from the compressed .tar.gz file. 
#This should give you a set of directories, labelled A through J.
#-------------------------------------------------------------------------------------------
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#-------------------------------------------------------------------------------------------
#Problem 1
#Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an 
#image of a character A through J rendered in a different font. Display a sample of the images 
#that we just downloaded. Hint: you can use the package IPython.display.

image1 = 'notMNIST_large\\A\\a29ydW5pc2hpLnR0Zg==.png'
img1 = ndimage.imread(image1) #Read the image
#Below command turns on inline plotting for IPython, where plot graphics will appear in your notebook.
#%matplotlib inline

#display the image without scaling
plt.imshow(img1)

#Scale and center the data
pixel_depth = 255.0
img1 = (img1 - pixel_depth/2)/pixel_depth
#Display the image after scaling
plt.imshow(img1)

#Result: Both the images look same

#-------------------------------------------------------------------------------------------
#Now let's load the data in a more manageable format. Since, depending on your computer setup you 
#might not be able to fit it all in memory, we'll load each class into a separate dataset, store 
#them on disk and curate them independently. Later we'll merge them into a single dataset of 
#manageable size.
#We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, 
#normalized to have approximately zero mean and standard deviation ~0.5 to make training easier 
#down the road.
#A few images might not be readable, we'll just skip them.
#-------------------------------------------------------------------------------------------
image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder) #all-files-names.png
    dataset = np.ndarray(shape = (len(image_files), image_size, image_size), dtype = np.float32)
    print(folder) #notMNIST_large\\A\\
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image) #notMNIST_large\\A\\a29ydW5pc2hpLnR0Zg==.png
        try:
            image_data = ndimage.imread(image_file).astype(float) #Read the image file as 28 X 28 numeric array
            image_data = (image_data - (pixel_depth/2))/(pixel_depth) #normalizing the data to have zero mean and small standard deviation
            if image_data.shape != (image_size, image_size): #every image should be 28 X 28 size
                raise Exception('Unexpected Image Shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('could not read:', image_file, ':', e, '-its ok, skipping.')
            
    dataset = dataset[0:num_images, :, :] 
    if (num_images < min_num_images):
        raise Exception('Many Fewer images than expected: %d < %d' %(num_images, min_num_images))
    
    print('Full dataset Tensor', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard Deviation:', np.std(dataset))
    return dataset  # return the dataset containing the images converted into 28 X 28 arrays
    
def maybe_pickle(data_folders, min_num_images_per_class, force = False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping Pickling' % set_filename)
        else:
            print('Pickling %s' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
                
    return dataset_names
    
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets  = maybe_pickle(test_folders, 1800)
#-------------------------------------------------------------------------------------------------
#Problem 2

#Let's verify that the data still looks good. Displaying a sample of the labels and images from 
#the ndarray. Hint: you can use matplotlib.pyplot.

#Now, to retrieve a normalized image, we need to read back a pickle file (there is one per 
#dataset folder or letter) into a tensor and grab a slice from it. 

pickle_file = train_datasets[0] # index 0 should be all As, 1 = all Bs, etc.
with open(pickle_file, 'rb') as f:
    letter_set = pickle.load(f)  # unpickle
    sample_idx = np.random.randint(len(letter_set))  # pick a random image index
    sample_img = letter_set[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_img)
#-------------------------------------------------------------------------------------------------
#Problem 3
#Another check: we expect the data to be balanced across classes. Verify that.

letter_len = []
for label, pickle_file in enumerate(train_datasets):
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        letter_len.append(len(letter_set))

print(letter_len)
#Ans = [52909, 52911, 52912, 52911, 52912, 52912, 52912, 52912, 52912, 52911]

#-------------------------------------------------------------------------------------------------
#Merge and prune the training data as needed. Depending on your computer setup, you might not be
#able to fit it all in memory, and you can tune train_size as needed. The labels will be stored 
#into a separate array of integers 0 through 9.
#-------------------------------------------------------------------------------------------------

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype = np.float32)
        labels = np.ndarray(nb_rows, np.float32)
    else:
        dataset, labels = None, None
        
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size = 0): #valid_size = validation_size
    
    num_classes = len(pickle_files)
    
    valid_dataset, valid_labels = make_arrays(valid_size, image_size) #Validation dataset & labels
    train_dataset, train_labels = make_arrays(train_size, image_size) #training dataset & labels
    
    vsize_per_class = valid_size//num_classes #total validation size/number of classes
    tsize_per_class = train_size//num_classes
    
    start_v, start_t = 0, 0
    end_v  , end_t   = vsize_per_class, tsize_per_class
    end_1            = vsize_per_class + tsize_per_class
    
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f) #load all the data in each class
                                            #"A" class loads 52909 images etc
                #lets shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                                
                if valid_dataset is not None:
                    #Load validation dataset, first vsize_per_class images will be for validation
                    valid_letter = letter_set[:vsize_per_class, :, :] #extract first vsize_per_class number of images into validation set
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    #Move the index to next available indexes in the validation dataset
                    start_v += vsize_per_class #change the starting index for next letter
                    end_v   += vsize_per_class #change the ending index for next letter
                    #always, end_v - start_v = vsize_per_class
                
                train_letter = letter_set[vsize_per_class:end_1, :, :] #load the next data after validation data, for a size of tsize_per_class
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                #Move the indexes to next available indexes in training dataset
                start_t += tsize_per_class
                end_t   += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
            
    return valid_dataset, valid_labels, train_dataset, train_labels
    
#----------------------------------------------------------------------------------------------
train_size = 200000
valid_size = 10000
test_size  = 10000

#load validation data and training datasets with labels
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)

#Load test data with labels
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size, 0)

print('Training', train_dataset.shape, train_labels.shape)
print('Validation', valid_dataset.shape, valid_labels.shape)
print('Test', test_dataset.shape, test_labels.shape)

#------------------------------------------------------------------------------------------
#Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
test_dataset,  test_labels  = randomize(test_dataset, test_labels)

#--------------------------------------------------------------------------------------------
# Finally, let's save the data for later reuse:

pickle_file = 'notMNIST.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset, 
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,        
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)    
    f.close()
except Exception as e:
    print('unable to save data to', pickle_file, ':', e)
    raise
#--------------------------------------------------------------------------------------------

stat_info = os.stat(pickle_file)    
print('Compressed Pickle size:', stat_info.st_size)

#---------------------------------------------------------------------------------------------
#building a logistic regression model
#---------------------------------------------------------------------------------------------
#Logistic regression model on 50 rows
y = train_labels[0:50]

#train_dataset.shape = (200000, 28, 28)
train1 = train_dataset[0:50, :, :]
#train1.shape = (50, 28, 28)
#Logistic regression cannot be built on a 3 Dimensional arrary. So reducing the dimensions to 2.
train1 = train1.reshape(50, 28*28)
#train1.shape = (50, 784)

#Build model
model1_50 = LogisticRegression()
model1_50 = model1_50.fit(train1, y)

#check the model performance on training data
model1_50.score(train1, y)
#ans = 1.0

#check the accuracy on validation dataset
validation_x = valid_dataset[0:50, :, :]
validation_x = validation_x.reshape(50, 28*28)
validation_y = valid_labels[0:50]

predicted = model1_50.predict(validation_x)
metrics.accuracy_score(validation_y, predicted) # = 0.34000000000000002

#---------------------------------------------------------------------------------------------
#building a logistic regression model with 1000 samples
#---------------------------------------------------------------------------------------------
#Logistic regression model on 1000 rows
y = train_labels[0:1000]

#train_dataset.shape = (200000, 28, 28)
train1 = train_dataset[0:1000, :, :]
#train1.shape = (1000, 28, 28)
#Logistic regression cannot be built on a 3 Dimensional arrary. So reducing the dimensions to 2.
train1 = train1.reshape(1000, 28*28)
#train1.shape = (1000, 784)

#Build model
model2_1000 = LogisticRegression()
model2_1000 = model2_1000.fit(train1, y)

#check the model performance on training data
model2_1000.score(train1, y)
#ans = 0.997

#check the accuracy on validation dataset
validation_x = valid_dataset[0:1000, :, :]
validation_x = validation_x.reshape(1000, 28*28)
validation_y = valid_labels[0:1000]

predicted = model2_1000.predict(validation_x)
metrics.accuracy_score(validation_y, predicted) #ans = 0.77100000000000002

#---------------------------------------------------------------------------------------------
#building a logistic regression model with entire samples
#---------------------------------------------------------------------------------------------
#Logistic regression model on 1000 rows
y = train_labels

#train_dataset.shape = (200000, 28, 28)
train1 = train_dataset
#train1.shape = (200000, 28, 28)
#Logistic regression cannot be built on a 3 Dimensional arrary. So reducing the dimensions to 2.
train1 = train1.reshape(len(train_dataset), 28*28)
#train1.shape = (200000, 784)

#Build model
model3_all = LogisticRegression()
model3_all = model3_all.fit(train1, y)

#check the model performance on training data
model3_all.score(train1, y)
#ans = 

#check the accuracy on validation dataset
validation_x = valid_dataset
validation_x = validation_x.reshape(len(valid_dataset), 28*28)
validation_y = valid_labels

predicted = model3_all.predict(validation_x)
metrics.accuracy_score(validation_y, predicted) #ans = 




    
                
                
                
                
                
                
                
                
        

       
            
            
            
        




