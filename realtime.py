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
from os.path import join

class RealTime_Classifier():
    def __init__(self, model_name=None, window_size=30, cutoff_freq_high=0.1, cutoff_freq_low=0.01, apply_hpf=True, apply_lpf=False):
		# define tensorflow model
		if model_name == None:
			reg = 1e-3
			self.model = tf.keras.Sequential([ \
			Conv1D(16, kernel_size = 5, \
				strides=1, activation = 'relu', \
				kernel_regularizer = l2(reg), \
				bias_regularizer = l2(reg), \
				input_shape=(window_size, 3)), \
			Flatten(), \
			Dense(8, activation='relu', \
				  kernel_regularizer=l2(reg), \
				  bias_regularizer=l2(reg)), \
			Dense(4, activation='relu', \
				  kernel_regularizer=l2(reg), \
				  bias_regularizer=l2(reg)), \
			Dense(2, activation='softmax', \
				  kernel_regularizer=l2(reg), \
				  bias_regularizer=l2(reg)) \
			])
		else:
			self.model = load_model('model.h5')
		
		self.apply_hpf = apply_hpf
		self.apply_lpf = apply_lpf		
        self.b_h, self.a_h = signal.butter(8, cutoff_freq_high, 'high')
        self.b_l, self.a_l = signal.butter(8, cutoff_freq_low, 'low')
		self.window_size = window_size
        self.X = np.zeros([self.window_size,3])
	
    def update(self, acc_reading):
        self.X = np.vstack([self.X[1:],np.array(acc_reading)])
	
	def train(self, files):
		# hyperparameter
		num_classes = len(files)
		
		# get data from each file
		X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
		for i, file in enumerate(files):
			X_raw = np.genfromtxt(join('data', file))
			train_val_split = int(2/3*len(X_raw))
			X_train, y_train = self._preprocess(X_raw[:train_val_split], self.window_size, i)
			X_val, y_val = self._preprocess(X_raw[train_val_split:], self.window_size, i)
		
		# concatenate the data
		X_train = np.vstack(X_train_list)
		y_train = np.hstack(y_train_list)
		X_val = np.vstack(X_val_list)
		y_val = np.hstack(y_val_list)
		
		# shuffle the data
		X_train, y_train = self._shuffle_data(X_train, y_train)
		X_val, y_val = self._shuffle_data(X_val, y_val)
		
		# filter the data
		for i, sample in enumerate(X_train):
			for j in range(sample.shape[-1]):
				if self.apply_hpf:
					X_train[i,:,j] = signal.filtfilt(self.b_h,self.a_h,sample[:,j])
				if self.apply_lpf:
					X_val[i,:,j] = signal.filtfilt(self.b_l,self.a_l,sample[:,j])
		
		# train model
		save_file = 'model.h5'
		optimizer = Adam()
		self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		self.model.summary()
		self.model.save(save_file)
		
		print("saved trained model to {}".format(save_file))

    def test(self):
		# filter the data
		if filte
        for i in range(self.X.shape[-1]):
			if self.apply_hpf:
				self.X[:,i] = signal.filtfilt(self.b_h, self.a_h, self.X[:,i])
			if self.apply_lpf:
				self.X[:,i] = signal.filtfilt(self.b_l, self.a_l, self.X[:,i])
			
		# get prediction
        probs = self.model.predict(np.array([self.X]), batch_size = 256)
        return probs

	def _preprocess(self, X_raw, window_size, label):
		# perform windowing
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
		indices = range(len(y))
		shuffled_indices = np.random.permutation(len(y))
		X[indices] = X[shuffled_indices]
		y[indices] = y[shuffled_indices]
		return X, y