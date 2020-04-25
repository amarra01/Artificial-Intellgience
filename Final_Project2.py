# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:18:01 2020

@author: armma
"""

import pandas 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras import models
from keras import layers 
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns

# In this code I am going to use keras to classify a binary output
#First we need to load the data scale it and define our training data
#Then I will define a network of layers that amps my input to my targets
#Then I will need a loss funtion and some kind of metric to understand the learning process
#Finally I will iterate through the training data using fit()

dataframe=pandas.read_csv('HTRU.csv',delim_whitespace=False, header='infer')
dataset=dataframe.values
print('dataset.shape',dataset.shape)
#Load the data from the HTRU csv and print its hsape

x = dataset[:,0:8]
y = dataset[:,8]
print(x)
print(y)
#Seperate the x variables and the y varaibles
np.random.seed(10)
X_MinMax = preprocessing.MinMaxScaler(x)
#scale the x variables
x = np.array(x).reshape((len(x),8))

y = np.array(y).reshape((len(y),1))
#reshape the x and y variables

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=42)
#here I define my training data and split it 20% off of it to create the test data



start_time=datetime.datetime.now()
#used to time the speed of my algotrithm
network=models.Sequential()
#.sequential is used for linear stacks of layers
network.add(layers.Dense(16, activation= 'relu', input_shape=(8,)))
#first dense layer 16-dimension activation relu input shape 8 for our varaibles
network.add(layers.Dense(64, activation= 'relu'))
#second layer 64 dimesnion activation relu
network.add(layers.Dense(1, activation = 'sigmoid'))
#third layer 1 dimension activation sigmoid turns values into 0,1 and output can be interpreted as probability
network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#I use binary_crossentropy for because of the single unit layer of a sigmoid function
network.summary()
    


history = network.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs = 250, batch_size = 50, verbose = 2) 
#the learning process takes numpy arrays of input data to the model via the fit()


#n_split=3
#for train_index, test_index in KFold(n_splits=n_split, random_state=None, shuffle=True).split(x):
#    x_train,x_test=x[train_index],x[test_index]
#    y_train,y_test=y[train_index],y[test_index]
#    network=create_model()
#    history = network.fit(X_train, Y_train, validation_data = (X_test,Y_test), epochs = 50, batch_size = 100, verbose = 2)





stop_time = datetime.datetime.now()
print ("Time required for optimization:",stop_time - start_time)  






#additional 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#These are plots to see how well the model is working and whether it is overfitting the training data



















 

