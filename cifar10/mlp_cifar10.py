import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as st
import pandas as pd

np.random.seed(1337)
batch_size = 128
nb_classes = 5
nb_epoch = 2
rg = l2(l=1e-3)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck']

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# train with airplane and dog
indexes = [i for i, c in enumerate(y_train) if c not in [0,1,2,3,4]]

X_train = X_train.reshape(50000, 3*32*32)
X_test = X_test.reshape(10000, 3*32*32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = np.delete(X_train, indexes, axis=0)
y_train = np.delete(y_train, indexes, axis=0)
y_train = y_train / 5.0
#y_train = np_utils.to_categorical(y_train, nb_classes)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(512, activation='relu', W_regularizer=rg, 
                input_shape=(3*32*32,)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', W_regularizer=rg))
#model.add(Dense(nb_classes, activation='softmax', W_regularizer=rg))

#sgd = SGD(lr=0.01, momentum=0.9);
model.compile(loss='binary_crossentropy', optimizer='sgd')
#model.compile(loss='categorical_crossentropy', optimizer='sgd')
T = 50


def evaluate(X, l=2, p=0.5):
    probs = []
    for i in range(T):
        probs.append(model.predict(np.array(X)))
    pred_mean = np.mean(probs, axis=0)
    pred_variance = np.std(probs, axis=0)
    #tau = l**2 * (1 - p) / (2 * N * rg.l2)
    #pred_variance += tau**-1
    
    return pred_mean, pred_variance


def plot_class(cl):
    indexes = [i for i, c in enumerate(y_test) if c != cl]
    m, v = evaluate(np.delete(X_test, indexes, axis=0))
    
    entropy = st.entropy(m, base=2)  
    ax = pd.DataFrame(entropy).plot(title=("Entropy",cl), kind='hist')  
    fig = ax.get_figure()
    fig.savefig('entropy'+str(cl)+".png")       

    
    plt.figure("class"+str(cl))
    plt.title('CLASS ' + classes[cl])
    plt.hist(m, label="mean")
    plt.hist(v, label="std")
    plt.xlim(0, 1)
    plt.legend()
    plt.xlabel('Probabilities')
    plt.savefig('mean_prob_'+str(cl)+".png")
    plt.close()

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          show_accuracy=False)

#plot_class(0)
#plot_class(5)
for i in range(10):
    plot_class(i)