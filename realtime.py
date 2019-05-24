##########################################################
# Authors:
# Derek Xu (UCLA)
# Justin Chang
##########################################################

import numpy as np # efficient matrix operations
import tensorflow as tf # machine learning tools

import sklearn # data cleaning
from scipy import signal # signal processing
import matplotlib.pyplot as plt # visualizations
from os.path import join # directory access

# for machine learning model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import *
from tensorflow.keras.models import load_model

##########################################################
# RealTime_Classifier Object
# This class does:
# 1) Define the Neural Network model
# 2) Train the model
# 3) Test the model in real-time
##########################################################

class RealTime_Classifier():
    def __init__(self, model_name=None, window_size=200, cutoff_freq_high=0.01, \
                 cutoff_freq_low=0.5, num_classes=2, apply_hpf=True, apply_lpf=True):
        
        # define machine learning model (tensorflow computation graph)
        if model_name == None:
            # default machine learning model
            reg = 1e-3
            self.model = tf.keras.Sequential([ \
            Conv1D(16, kernel_size = 5, \
                strides=1, activation = 'relu', \
                kernel_regularizer = l2(reg), \
                bias_regularizer = l2(reg), \
                input_shape=(window_size, 3)), \
            Conv1D(8, kernel_size = 5, \
                strides=1, activation = 'relu', \
                kernel_regularizer = l2(reg), \
                bias_regularizer = l2(reg)), \
            Conv1D(4, kernel_size = 5, \
                strides=1, activation = 'relu', \
                kernel_regularizer = l2(reg), \
                bias_regularizer = l2(reg)), \
                        Flatten(), \
            Dense(8, activation='relu', \
                kernel_regularizer=l2(reg), \
                bias_regularizer=l2(reg)), \
            Dense(num_classes, activation='softmax', \
                kernel_regularizer=l2(reg), \
                bias_regularizer=l2(reg))])
        else:
            # load from disk
            self.model = load_model('model.h5')
        
        # define filter parameters
        self.apply_hpf = apply_hpf
        self.apply_lpf = apply_lpf
        self.b_h, self.a_h = signal.butter(8, cutoff_freq_high, 'high')
        self.b_l, self.a_l = signal.butter(8, cutoff_freq_low, 'low')
        
        # define model hyper-parameters
        self.window_size = window_size
        self.num_classes = num_classes
        
        # buffer for real-time prediction
        self.X = np.zeros([self.window_size,3])

    def train(self, files, epochs = 100, verbose = False):
        # check we have required # of files
        assert(self.num_classes == len(files))
        
        # each file contains one class (i.e. get data for each class)
        X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
        for i, file in enumerate(files):
            # retrieve raw data from each file
            X_raw = np.genfromtxt(join('data', file),delimiter=',')
            # define train validation split index
            train_val_split = int(2/3*len(X_raw))
            # split and pre-process the raw data
            X_train, y_train = self._preprocess(X_raw[:train_val_split], self.window_size, i)
            X_val, y_val = self._preprocess(X_raw[train_val_split:], self.window_size, i)
            # append training and testing data to accumulator
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_val_list.append(X_val)
            y_val_list.append(y_val)
        
        # concatenate the data along classes (i.e. X = [X_class1|X_class2|...|X_classN])
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        X_val = np.vstack(X_val_list)
        y_val = np.hstack(y_val_list)
        
        # shuffle the data (for training purposes)
        X_train, y_train = self._shuffle_data(X_train, y_train)
        X_val, y_val = self._shuffle_data(X_val, y_val)
        
        # filter the training data
        for i, sample in enumerate(X_train):
            display_flag = ((i == 0) and verbose) #define when we want to plot the data
            if display_flag:
                # display our pre-processed signal for the first training sample
                plt.figure()
                plt.subplot(211)
                for j in range(sample.shape[-1]):
                    plt.plot(X_train[i,:,j].tolist())
                plt.xlabel('time')
                plt.ylabel('acc')
                plt.title('Prefiltered')
            
            for j in range(sample.shape[-1]):
                # apply butter-worth filters on the signals
                if self.apply_hpf:
                    X_train[i,:,j] = signal.filtfilt(self.b_h,self.a_h,sample[:,j])
                if self.apply_lpf:
                    X_train[i,:,j] = signal.filtfilt(self.b_l,self.a_l,sample[:,j])

            if display_flag:
                # display our post-processed signal for the first training sample
                plt.subplot(212)
                for j in range(sample.shape[-1]):
                    plt.plot(X_train[i,:,j].tolist())
                plt.xlabel('time')
                plt.ylabel('acc')
                plt.title('Postfiltered')
                plt.show()

        # filter the validation data
        for i, sample in enumerate(X_val):
            for j in range(sample.shape[-1]):
                if self.apply_hpf:
                    X_val[i,:,j] = signal.filtfilt(self.b_h,self.a_h,sample[:,j])
                if self.apply_lpf:
                    X_val[i,:,j] = signal.filtfilt(self.b_l,self.a_l,sample[:,j])
        
        # define optimization method
        optimizer = SGD(lr=1e-3)

        # define loss function
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # train model
        self.model.summary()
        batch_size = 256
        history = self.model.fit(X_train, \
                        to_categorical(y_train), \
                        epochs=epochs, \
                        batch_size=batch_size, \
                        validation_data = (X_val, to_categorical(y_val)))

        # save the model
        save_file = 'model.h5'
        self.model.save(save_file)
        print("saved trained model to {}".format(save_file))

    def update(self, acc_reading):
        # update buffer for real-time prediction
        self.X = np.vstack([self.X[1:],np.array(acc_reading)])
    
    def test(self):
        # filter the buffer
        for i in range(self.X.shape[-1]):
            if self.apply_hpf:
                self.X[:,i] = signal.filtfilt(self.b_h, self.a_h, self.X[:,i])
            if self.apply_lpf:
                self.X[:,i] = signal.filtfilt(self.b_l, self.a_l, self.X[:,i])
            
        # generate prediction on the buffer
        probs = self.model.predict(np.array([self.X]), batch_size = 256)
        return probs

    def _preprocess(self, X_raw, window_size, label):
        # perform windowing
        # (our buffer is of size window_size, so we want to have each training sample be the same dimension)
        X = []
        y = []
        for i in range(X_raw.shape[0]-window_size):
                X.append(X_raw[i:i+window_size])
                y.append(label)
        X=np.array(X)
        y=np.array(y)
        return X, y

    def _shuffle_data(self, X, y):
        # shuffle the data
        # (our optimization method requires that the data be shuffled)
        indices = range(len(y))
        shuffled_indices = np.random.permutation(len(y))
        X[indices] = X[shuffled_indices]
        y[indices] = y[shuffled_indices]
        return X, y
