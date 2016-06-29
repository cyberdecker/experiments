import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
import theano
import scipy
from keras.layers import Activation, Dense, Dropout, ELU
from keras.regularizers import l2

theano.config.compute_test_value = 'off'

class PoorBayesian(Layer):

    def __init__(self, output_dim, mean_prior, std_prior, **kwargs):
        self.output_dim = output_dim
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        super(PoorBayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.W = K.random_normal(shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]))

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        self.W_sample = self.W*K.log(1.0 + K.exp(self.log_std)) + self.mean
        return K.dot(x, self.W_sample) + self.bias

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class MyDropout(Layer):

    def __init__(self, p, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(MyDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.dropout(x, level=self.p)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(MyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def log_gaussian(x, mean, std):
    return -K.log(2*np.pi)/2.0 - K.log(std) - (x-mean)**2/(2*std**2)

def log_gaussian2(x, mean, log_std):
    log_var = 2*log_std
    return -K.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*K.exp(log_var))


def bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs):
    def loss(y_true, y_pred):
        log_p = K.variable(0.0)
        log_q = K.variable(0.0)
        nb_samples = batch_size
        for layer in model.layers:
            if type(layer) is Bayesian:
                mean = layer.mean
                log_std = layer.log_std
                W_sample = layer.W_sample
                # prior
                log_p += K.sum(log_gaussian(W_sample, mean_prior, std_prior))/nb_samples
                # posterior
                log_q += K.sum(log_gaussian2(W_sample, mean, log_std))/nb_samples

        #log_likelihood = objectives.categorical_crossentropy(y_true, y_pred)
        log_likelihood = K.sum(log_gaussian(y_true, y_pred, std_prior))

        return K.sum((log_q - log_p)/nb_batchs - log_likelihood)/batch_size
    return loss

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

mean_prior = 0.0
std_prior = 0.05
in_dim = 1
out_dim = 1
batch_size = 64
nb_batchs = diabetes_X_train.shape[0]//batch_size

model = Sequential()
model.add(PoorBayesian(1, mean_prior, std_prior, input_shape=[in_dim]))
model.add(Activation('relu'))
#model.add(PoorBayesian(out_dim, mean_prior, std_prior))
model.add(Activation('linear'))
#loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)
#model.compile(loss=loss, optimizer='sgd', metrics=['accuracy'])

model.compile(loss='mean_squared_error', optimizer='sgd')

# how many parameters?
#model.summary()

model.fit(diabetes_X_train, diabetes_Y_train, nb_epoch=10000, batch_size=64,verbose=0)

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

#Make lines and plot for both
#w1, w0 = model.get_weights()

# layer 0 / layer 2 with weights, layer 1 and 3 with activation
w0 = model.layers[0].get_weights()[2] # bias
w1 = model.layers[0].get_weights()[0] # weight

print "w0", w0, "w1", w1



weights = np.ndarray(shape=(1,1))
count = 0
for layer in model.layers:
    weights = layer.get_weights()
    count += 1
    print count, weights

print "reg",regr.intercept_,regr.coef_

'''
w1 = weights[1]    
w0 = weights[0]  
'''  
#print "weights", weights.shape


tt = np.linspace(np.min(diabetes_X[:, 0]), np.max(diabetes_X[:, 0]), 10)
nn_line = w0+w1*tt
lreg_line = regr.intercept_+regr.coef_*tt 

plt.scatter(diabetes_X_test, diabetes_Y_test,  color='black')
#plt.plot(diabetes_X[:,0],diabetes['target'],'kx',tt,lreg_line,'r-',tt,nn_line[0],'y--')
plt.plot(tt,lreg_line,'r-',tt,nn_line[0],'b--')

plt.savefig("plot_bayesnn.png")
