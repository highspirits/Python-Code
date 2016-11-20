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
! pip install seaborn
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

threshold = 0.5
#identify highly correlated fields
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))




