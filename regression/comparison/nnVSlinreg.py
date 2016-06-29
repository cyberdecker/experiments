import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import theano
from keras.regularizers import l2

theano.config.compute_test_value = 'off'

print "Load dataset..."
# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

size_test = 50

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-size_test]
diabetes_X_test = diabetes_X[-size_test:]
print "X train shape", diabetes_X_train.shape

# Split the targets into training/testing sets
diabetes_Y_train = diabetes.target[:-size_test]
diabetes_Y_test = diabetes.target[-size_test:]
print "Y train shape", diabetes_Y_train.shape

###Linear Regression with sklearn
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_Y_train)


###Linear Regression with keras
# Initialize Network
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1))
model.add(Activation('relu'))
#model.add(Dense(input_dim=1, output_dim=1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='sgd')

# how many parameters?
#model.summary()

model.fit(diabetes_X_train, diabetes_Y_train, nb_epoch=5000, batch_size=64,verbose=0)

# Uncertainity
reg = l2(l=0.001)
p=0.5
l=1.0
tau = l**2 * (1-p)/(2*diabetes_X_train.shape[0]*reg.l2)

test_pred_mean = []
test_pred_var = []
test_pred_std = []

for i, x in enumerate(diabetes_X_test):
    probs = model.predict(np.array([x]*10), batch_size=1)
    pred_mean = probs.mean(axis=0)[0]
    pred_std = probs.std(axis=0)
    pred_var = probs.var(axis=0)[0]
    
    pred_var += tau**-1
    test_pred_mean.append(pred_mean)
    test_pred_std.append(pred_std.mean())
    test_pred_var.append(pred_var)

'''
plt.figure()
plt.hist(test_pred_mean)
plt.figure()
plt.hist(test_pred_std)
plt.figure()
plt.hist(test_pred_var)
'''
#plt.fill_between(t, lower_bound, upper_bound, facecolor='green', alpha=0.5)

#Make lines and plot for both

# layer 0 / layer 2 with weights, layer 1 and 3 with activation
w0 = model.layers[0].get_weights()[1] # bias
w1 = model.layers[0].get_weights()[0] # weight

weights = np.ndarray(shape=(1,1))
for layer in model.layers:
    weights = layer.get_weights()
    #print weights

#print "reg",regr.intercept_,regr.coef_

'''
w1 = weights[1]    
w0 = weights[0]  
'''  
#print "weights", weights.shape


tt = np.linspace(np.min(diabetes_X[:, 0]), np.max(diabetes_X[:, 0]), 10)
nn_line = w0+w1*tt
lreg_line = regr.intercept_+regr.coef_*tt 

plt.figure()
plt.scatter(diabetes_X_test, diabetes_Y_test,  color='black')
plt.plot(tt,lreg_line,'r-',tt,nn_line[0],'b--')

plt.savefig("plot_nn.png")
