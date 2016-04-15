# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:33:10 2016

@author: giseli
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.engine.topology import Layer
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.datasets import mnist
import keras.backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import pandas as pd
import seaborn as sns
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import scipy.stats as st

class MyDropout(Layer):
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
N = 50

def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', W_regularizer=rg, input_shape=(784,)))
    model.add(MyDropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', W_regularizer=rg))
    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
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
    
def evaluate(X, y, cl, my_range):
    probs = []
    classes = []
    for i in range(N):
        predictions = model.predict(np.array(X))
        #print "Shape predictions", predictions.shape
        probs.append(predictions)

    pred_mean = np.mean(probs, axis=0)
    pred_std = np.std(probs, axis=0)
    
    entropy = st.entropy(pred_mean)    
    ax = pd.DataFrame(entropy).plot(title=("Entropy",cl), kind='hist')  
    fig = ax.get_figure()
    fig.savefig('entropy'+str(cl)+".png") 
    
    return pred_mean, pred_std, classes    
    
def plot_class(cl, X_test, y_test):
    indexes = [i for i, c in enumerate(y_test) if c != cl]
    test_class = np.delete(X_test, indexes, axis=0)  
    y = np.delete(y_test, indexes, axis=0)
    test_class = np.delete(X_test, indexes, axis=0)    
    # Test only 10 samples of the current class
    #my_range = 100
    # Test on all
    my_range = test_class.shape[0]
    m, s, classes = evaluate(test_class[0:my_range], y, cl, my_range)
        
    # Plot std    
    plt.figure('All_Outputs_SD' + str(cl))
    plt.title('All Outputs SD #' + str(cl))
    plt.hist(s)
    plt.xlim(0, 0.5)
    plt.xlabel('Standard Deviation')
    #plt.show()
    plt.savefig('std'+str(cl)+".png")
    plt.close() 
    # Plot mean

    plt.figure('All_Outputs_Mean' + str(cl))
    plt.title('All Outputs Mean #' + str(cl))
    plt.hist(m)
    plt.xlim(0, 1)
    plt.xlabel('Mean')
    #plt.show()
    plt.savefig('mean'+str(cl)+".png")
    plt.close()

    
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#indexes08 = [i for i, c in enumerate(y_train) if c < 9]
#test_indexes08 = [i for i, c in enumerate(y_test) if c < 9]

index9 = [i for i, c in enumerate(y_train) if c >= 9]
test_index9 = [i for i, c in enumerate(y_test) if c >= 9]

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Exclude the 9 digit from training
X_train = np.delete(X_train, index9, axis=0)
y_train = np.delete(y_train, index9, axis=0)

#print(X_train.shape, 'train samples')
#print(X_test.shape, 'test samples')

train_size = X_train.shape[0]

y_train = np_utils.to_categorical(y_train, nb_classes)

# Create model
print ("Creating model...")
model = create_model()  

# fit model to training data:
history = LossHistory()
model.fit(X_train, y_train, 
          batch_size, 
          nb_epoch=nb_epoch,
          verbose=1, show_accuracy=True,
          callbacks=[history])
 
#plt.figure()
#plt.hist(history.losses)

#plt.show()
#plt.close()


#print "History loss shape ", history.losses
#ax = pd.DataFrame(entropy).plot(title=("entropy"), kind='hist')  
#fig = ax.get_figure()
#fig.savefig('hist_entropy'+".png")  

#plot_class(9, X_test, y_test)
for i in range(10):
    plot_class(i, X_test, y_test)