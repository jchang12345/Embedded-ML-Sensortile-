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

% for how to adapt the input to predict for real time:
%X = np.zeros([30,3])
%X = np.vstack([X[1:],np.array([accelx,accely,accelz])])

class RT_MLEngine():
    def __init__(self):
        self.model = load_model('model.h5')
        self.b_h, self.a_h = signal.butter(8, 0.1, 'high')
        self.X = np.zeros([30,3])
    def update(acc_reading):
        self.X = np.vstack([self.X[1:],np.array(acc_reading)])
    def predict():
        for i in range(self.X.shape[-1]):
            self.X[:,i] = signal.filtfilt(self.b_h, self.a_h, self.X[:,i])
        results = self.model.predict(np.array([self.X]), batch_size = 1)

        classes = np.argmax(results, axis=1)
        temp = collections.Counter(classes)

        chance_0 = temp[0]/len(classes)
        chance_1 = temp[1]/len(classes)

        return chance_0, chance_1
