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
import matplotlib.mlab as mlab

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
nb_classes = 9
nb_epoch = 2
rg = l2(l=1e-3)
T = 50

def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', W_regularizer=rg, input_shape=(784,)))
    model.add(MyDropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', W_regularizer=rg))
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

def evaluate(X):
    probs = []
    for i in range(T):
        probs.append(model.predict(np.array(X)))
    pred_mean = np.mean(probs, axis=0)
    pred_std = np.std(probs, axis=0)
    return pred_mean, pred_std
    
def plot_class(cl, X_test, y_test):
    indexes = [i for i, c in enumerate(y_test) if c != cl]
    test_class = np.delete(X_test, indexes, axis=0)
    #print "Shape of test:", test_class.shape
    m, s = evaluate(test_class)
    plt.figure('All_Outputs_SD' + str(cl))
    plt.title('All Outputs SD #' + str(cl))
    plt.hist(s)
    plt.xlim(0, 0.5)
    plt.xlabel('Standard Deviation')
    plt.figure('All_Outputs_Mean' + str(cl))
    plt.title('All Outputs Mean #' + str(cl))
    plt.hist(m)
    plt.xlim(0, 1)
    plt.xlabel('Mean')
    
    # example data
    mu = np.mean(m) # mean of distribution
    sigma = np.mean(s) # standard deviation of distribution
    x = mu + sigma * m
    num_bins = 50
    # the histogram of the data
    plt.figure('PDF' + str(cl))
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)    
    plt.plot(bins, y, 'r--')
    #plt.xlabel('Smarts')
    #plt.ylabel('Probability')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    
# the data, shuffled and split between train and test sets
# For training, get the digits from 0 to 8
X_train, y_train = load_mnist("training", asbytes=False, digits=[0,1,2,3,4,5,6,7,8]) #X_test, y_test = load_mnist('testing', digits=[0,1])
# For testing, get the digit 9
X_test9, y_test9 = load_mnist('testing', digits=[9])
# For testing, get all digits
X_test, y_test = load_mnist('testing')

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
image = tile_raster_images( X=X_train[idx],
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
y = np.zeros([size, nb_classes])
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
          callbacks=[history]) #validation_data=(X_test, y_test)
          
# show the SGD progress:
plt.figure()
plt.plot(movingaverage(history.losses, 20))
plt.show()

'''
score = model.evaluate(X_val, y_val,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''

tp = model.predict(X_val)
pred = np.array( [np.argmax( tp[i] ) for i in range(len(tp))] )
ct = pd.crosstab(y_val, pred, rownames=["Actual"], colnames=["Predicted"], margins=False); 
# the overall accuracy:
print("Test accuracy:", sum( np.diagonal(ct) )/X_val.shape[0])        

image_array = tile_raster_images(
                         X=model.get_weights()[0].transpose(),
                         img_shape=(28,28), 
                         tile_shape=(11,12),
                         tile_spacing=(1,1))
plt.figure(figsize=(14,14))
plt.imshow(image_array, cmap='Greys', aspect='equal', interpolation='bilinear')  

for i in range(10):
    plot_class(i, X_test, y_test)
    
