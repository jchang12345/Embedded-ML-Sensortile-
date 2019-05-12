import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn

import collections
import tensorflow as tf
from scipy import signal
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import *
from tensorflow.keras.models import load_model

def preprocess(X_raw, window_size, label):
    # perform windowing
    X = []
    y = []
    for i in range(X_raw.shape[0]-window_size):
            X.append(X_raw[i:i+window_size])
            y.append(label)
    X=np.array(X)
    y=np.array(y)
    return X, y

def shuffle_data(X, y):
    # shuffle the data
    indices = range(len(y))
    shuffled_indices = np.random.permutation(len(y))
    X[indices] = X[shuffled_indices]
    y[indices] = y[shuffled_indices]
    return X, y

# load the data from .csv
X_raw_1 = np.genfromtxt('data/acc_data_1_test.csv', delimiter=',')
X_raw_2 = np.genfromtxt('data/acc_data_2_test.csv', delimiter=',')

# look at the shapes of X_raw
print(X_raw_1.shape)
print(X_raw_2.shape)

window_size = 30
X_1, y_1 = preprocess(X_raw_1, window_size, 0)
X_2, y_2 = preprocess(X_raw_2, window_size, 1)

#import matplotlib.pyplot as plt
#b_l,a_l = signal.butter(8, 0.01, 'low')
b_h,a_h = signal.butter(8, 0.1, 'high')
for i, sample in enumerate(X_1):
    for j in range(sample.shape[-1]):
        #temp = signal.filtfilt(b_l,a_l,sample[:,j])
        X_1[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])
for i, sample in enumerate(X_2):
    for j in range(sample.shape[-1]):
        #temp = signal.filtfilt(b_l,a_l,sample[:,j])
        X_2[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])

# load tensorflow model
model = load_model('model.h5')
results = model.predict(X_1, batch_size=256)

classes = np.argmax(results, axis=1)
temp = collections.Counter(classes)

chance_0 = temp[0]/len(classes)
chance_1 = temp[1]/len(classes)

print('class 1')
print('probability of class 0: {}'.format(chance_0))
print('probability of class 1: {}'.format(chance_1))

results = model.predict(X_2, batch_size=256)

classes = np.argmax(results, axis=1)
temp = collections.Counter(classes)

chance_0 = temp[0]/len(classes)
chance_1 = temp[1]/len(classes)

print('class 2')
print('probability of class 0: {}'.format(chance_0))
print('probability of class 1: {}'.format(chance_1))
