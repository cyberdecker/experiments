import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

print "Load dataset..."
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

# Linear Regression with sklearn
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_Y_train)

# Linear Regression with keras
model = Sequential()
model.add(Dense(1, input_dim=1,init='uniform'))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
model.compile(loss='mse', optimizer='sgd')

'''
model.add(Dense(2, activation='linear', input_dim=1))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
sgd = SGD(lr=0.01, momentum=0.9)
#model.compile(loss='binary_crossentropy', optimizer=sgd)
#model.compile(loss='mean_absolute_error', optimizer='rmsprop')
model.compile(loss='mean_squared_error', optimizer='rmsprop')
'''
# how many parameters?
model.summary()
model.fit(diabetes_X_train, diabetes_Y_train, nb_epoch=1000, batch_size=64,verbose=0)

# Get lines and plot the two regressions
w1, w0 = model.get_weights()
# layer 0 / layer 2 with weights, layer 1 and 3 with activation
#w0 = model.layers[0].get_weights()[1] # bias
#w1 = model.layers[0].get_weights()[0] # weight
#print w0
#print w1

'''
weights = np.ndarray(shape=(1,1))
for layer in model.layers:
    weights = layer.get_weights()
w1 = weights[1]    
w0 = weights[0]  
'''  
#print "weights", weights.shape

tt = np.linspace(np.min(diabetes_X[:, 0]), np.max(diabetes_X[:, 0]), 10)
nn_line = w0+w1*tt
lreg_line = regr.intercept_+regr.coef_*tt 

# Plot the results
plt.scatter(diabetes_X_test, diabetes_Y_test,  color='black')
plt.plot(diabetes_X[:,0],diabetes['target'],'kx',tt,lreg_line,'r-',tt,nn_line[0],'b--')
plt.savefig("plot.png")
plt.show()
