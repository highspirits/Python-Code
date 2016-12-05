# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 23:43:26 2016

@author: vsnalam
"""

import os
os.chdir("C:\Sreedhar\Python\Code")

from sklearn.datasets import load_iris
import numpy as np
#from sklearn.preprocessing import train_test_split
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn import linear_model

iris = datasets.load_iris()
#Build linear regression
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 999)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
regr.coef_   #Print linear regression coefficients
y_test_predict = regr.predict(x_test)
regr_mse = np.mean((y_test_predict - y_test)**2) #Mean Square error value


#Ordinary Least Squares model
#Read airfoil data from UCI
import pandas as pd
#Link for airfoil data - https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat
airfoil = pd.read_excel("C:\\Sreedhar\\Python\\Airfoil.xlsx")
#Exploratory Analysis
airfoil.dtypes
airfoil.describe()
airfoil.info()

#x = data, y = target
x = airfoil.ix[:, 0:4]
y = airfoil.ix[:, 5]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.7)
lreg = linear_model.LinearRegression()
lreg.fit(xtrain, ytrain)
ypredict = lreg.predict(xtest)

#MSE calculation
np.mean((ypredict - ytest) ** 2)


