# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:33:10 2016

@author: giseli
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, MaskedLayer, Activation, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import matplotlib.pyplot as plt
import keras.callbacks as cb
import PIL.Image
from utils import tile_raster_images 
from keras.callbacks import Callback
import pandas as pd
from load_mnist import load_mnist

class MyDropout(MaskedLayer):
    def __init__(self, p, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.p = p
    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            X = K.dropout(X, level=self.p)
        return X
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'p': self.p}
        base_config = super(MyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

np.random.seed(1337)
batch_size = 128
nb_classes = 10
nb_epoch = 2
rg = l2(l=1e-3)

def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', W_regularizer=rg, input_shape=(784,)))
    model.add(MyDropout(0.5))
    model.add(Dense(10, activation='softmax', W_regularizer=rg))
    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    # how many parameters?
    model.summary()
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# show the SGD progress:
def movingaverage(x, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(x, window, 'same')

# the data, shuffled and split between train and test sets
X_train, y_train = load_mnist("training", asbytes=False) #X_test, y_test = load_mnist('testing', digits=[0,1])
X_test, y_test = load_mnist('testing') #, digits=[0,1])
# Get the last 2000 for validation
X_train, X_val = X_train[:-2000], X_train[-2000:]
y_train, y_val = y_train[:-2000], y_train[-2000:]

train_size = X_train.shape[0]
X_train = X_train.reshape(train_size, 784)
test_size = X_test.shape[0]
X_test = X_test.reshape(test_size, 784)
X_val = X_val.reshape(2000, 784)

print "X shape", X_train.shape

idx = np.random.randint(train_size, size=9)
idx.sort()
image = tile_raster_images(
                         X=X_train[idx],
                         img_shape=(28, 28), 
                         tile_shape=(3,3),
                         tile_spacing=(1, 1))
plt.figure()                         
plt.imshow(image, cmap='Greys')
print(y_train[idx])
plt.figure()
plt.hist(y_train, range=(-0.5,9.5), rwidth=0.8, facecolor='green')
#plt.imshow() 

# reshape training data:
size = y_train.shape[0]
y = np.zeros([size, 10])
for i in range(size): y[i, y_train[i]] = 1

# Create model
print ("Creating model...")
model = create_model()  

# fit model to training data:
history = LossHistory()
model.fit(X_train, y, 
          batch_size=20, 
          nb_epoch=2,
          verbose=1, 
          callbacks=[history])
          
# show the SGD progress:
plt.figure()
plt.plot(movingaverage(history.losses, 20))
plt.show()

tp = model.predict(X_val)
pred = np.array( [np.argmax( tp[i] ) for i in range(len(tp))] )
ct = pd.crosstab(y_val, pred, rownames=["Actual"], colnames=["Predicted"], margins=False); 

# the overall accuracy:
print(sum( np.diagonal(ct) )/10000.0)        

image_array = tile_raster_images(
                         X=model.get_weights()[0].transpose(),
                         img_shape=(28,28), 
                         tile_shape=(11,12),
                         tile_spacing=(1,1))
plt.figure(figsize=(14,14))
plt.imshow(image_array, cmap='Greys', aspect='equal', interpolation='bilinear')  