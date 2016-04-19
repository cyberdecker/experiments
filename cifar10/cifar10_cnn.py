# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:28:34 2016

@author: giseli
"""

from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 5
nb_epoch = 1
data_augmentation = False
p = 0.20 # percentage of dataset to be used in transfer learning
np.random.seed(1337)  # for reproducibility

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, y_test) = cifar10.load_data()

# create two datasets one with classes below 5 and one with 5 and above
X_train_A = X_train[Y_train[:,0] < 5]
Y_train_A = Y_train[Y_train[:,0] < 5]
X_test_A = X_test[y_test[:,0] < 5]
Y_test_A = y_test[y_test[:,0] < 5]

X_train_B = X_train[Y_train[:,0] >= 5]
Y_train_B = Y_train[Y_train[:,0] >= 5] - 5  # make classes start at 0 for
X_test_B = X_test[y_test[:,0] >= 5]                          # np_utils.to_categorical
Y_test_B = y_test[y_test[:,0] >= 5] - 5

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

all_layers = [
    Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', input_shape=(img_channels, img_rows, img_cols)),
    Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'),
    MaxPooling2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Convolution2D(nb_filters*2, nb_conv, nb_conv, border_mode='same',activation='relu'),
    Convolution2D(nb_filters*2, nb_conv, nb_conv, activation='relu'),
    MaxPooling2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(nb_classes, activation='softmax')
]

def downsample(p, X, Y):
    print ('Original sample size: ', X.shape[0])
    print ('Using p: ', p)
    ids = np.random.permutation(X.shape[0])
    n_samples = int(p*X.shape[0])
    new_X = X[ ids[0:n_samples], :]
    new_Y = Y[ ids[0:n_samples], :]
    print ('New sample size: ', new_X.shape[0])     
    return new_X, new_Y
    
def train_model(model, train, test, nb_classes, suffix=""):
    #X_train = train[0].reshape(train[0].shape[0], 1, img_rows, img_cols)
    #X_test = test[0].reshape(test[0].shape[0], 1, img_rows, img_cols)
    X_train = train[0]
    X_test = test[0]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(train[1], nb_classes)
    Y_test = np_utils.to_categorical(test[1], nb_classes)
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    t = datetime.now()
    print('Fit model')
    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=False, verbose=1,
              validation_split=0.2)
    print('Training time: %s' % (datetime.now() - t))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights("trained_modelB_freeze_%d.hdf5"%(suffix), overwrite=True)
    return score
        
def func_train_A(model):
 
    # train model for A split
    print ('Training for A')
    train_model(model,
            (X_train_A, Y_train_A),
            (X_test_A, Y_test_A), nb_classes )
    model.save_weights("trained_modelA.hdf5", overwrite=True)
    return model
        
if __name__=="__main__":

    do_train_A = True # train A or use previously saved hdf5
    
    model = Sequential()
    for l in all_layers:
        model.add(l)          
         
    #X_train_B, Y_train_B = downsample(p, X_train_B, Y_train_B)
    
    if do_train_A:     
        model = func_train_A(model)
    else:
        #model.load_weights("trained_modelA.hdf5")
        print ("Load something")
        
    number_of_layers = len(model.layers)
    scores = np.zeros( (number_of_layers, 2) )