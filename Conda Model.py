# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:18:12 2016

@author: vsnalam
"""

import os
os.chdir("C:\Sreedhar\Data Science Competitions\Kaggle Competitions\Allstate claims severity\Data\Code")

import pandas as pd
#read data
train = pd.read_csv("C:\\Sreedhar\\Data Science Competitions\\Kaggle Competitions\\Allstate claims severity\\Data\\train\\train.csv")
test = pd.read_csv("C:\\Sreedhar\\Data Science Competitions\\Kaggle Competitions\\Allstate claims severity\\Data\\test\\test.csv")

#store test ID and remove from test
test.id = test['id']
test.drop('id', 1, inplace=True)

#set option to display max rows and cols
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

#drop id column from train
train.drop('id', 1, inplace=True)

#Some basic data analysis
train.shape 
train.describe()
train.skew() #Values close to zero mean less skew. 

#import few libraries 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#"loss" is skewed to a large extent
sns.violinplot(data=train, y='loss')
#correcting the skewness of "loss" variable
train['loss'] = np.log1p(train['loss'])
sns.violinplot(data=train, y='loss')

#find correlation between columns. Correlation can be done on continuous data only. 
train_corr = train.corr()
#size of train_corr is 116 columns

size = 116
threshold = 0.5
split= 117
cols = train.columns

corr_list = []
#identify highly correlated fields
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (train_corr.iloc[i,j] >= threshold and train_corr.iloc[i,j] < 1) or (train_corr.iloc[i,j] < 0 and train_corr.iloc[i,j] <= -threshold):
            corr_list.append([train_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

labels = []
#All the algorithms require numerical data. Converting strings to numerical data using one-hot encoding
#set() gives the uniquq values in a list
for i in range(0, 117):
    train_unique = train[cols[i]].unique()
    test_unique  = test[cols[i]].unique()
    labels.append(list(set(train_unique) | set(test_unique)))

#-----------------------------------------------------------------------------------------
#Untested Code - start
#-----------------------------------------------------------------------------------------
#one hot encoding of categorical attributes
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(train.iloc[:,i])
    feature = feature.reshape(train.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)
    
#Dummyfying the categorical data
# Make a 2D array from a list of 1D arrays
encoded_cats = np.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
train_encoded = np.concatenate((encoded_cats,train.iloc[:,split:].values),axis=1)
#-----------------------------------------------------------------------------------------
#Untested Code - end
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
#Converting Categorical data to dummy data - start
#-----------------------------------------------------------------------------------------
# step - 1: Add a column which tells the dataset is test or train - this column should be numeric
train['cat_type'] = 1
test['cat_type'] = 2

#Merge train and test data 
tt = pd.concat([train, test])

#Extract columns which are categorical (do not include 'id' column) 1:117
col_names = tt.colnames
tt_catonly = tt[col_names[1:117]]

#dummify the data 
tt_catonly_dummy = pd.get_dummies(tt_catonly, drop_first = True)

#remove the categorical columns from merged test+train dataset
cat_cols = tt_catonly.columns
tt.drop(cat_cols, axis = 1, inplace = True)

#add dummified columns in the merged dataset 
tt_final = pd.concat([tt, tt_catonly_dummy], axis = 1)

#split merged dataset into test and train
train_new = tt_final.query('cat_type == 1')
test_new = tt_final.query('cat_type == 2')

#drop loss column from test
del test_new['loss']

#drop new column added to differentiate test vs train (cat_type)
del train_new['cat_type']
del test_new['cat_type']

#delete un-wanted variables
del test, train, tt, tt_catonly, tt_catonly_dummy

#-----------------------------------------------------------------------------------------
#Converting Categorical data to dummy data - end
#-----------------------------------------------------------------------------------------









