# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:28:02 2016

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

#Some basic data analysis
train.shape 
train.describe

#drop id column from train
train.drop('id', 1, inplace=True)