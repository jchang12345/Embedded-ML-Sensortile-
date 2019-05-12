import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn

import tensorflow as tf
from scipy import signal
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import *

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

def gen_model():
    # load the data from .csv
    X_raw_1 = np.genfromtxt('data/acc_data_1.csv', delimiter=',')
    X_raw_2 = np.genfromtxt('data/acc_data_2.csv', delimiter=',')

    # look at the shapes of X_raw
    print('Raw Shape:')
    print(X_raw_1.shape)
    print(X_raw_2.shape)

    # split the data
    window_size = 30
    train_val_split = int(2/3*len(X_raw_1))
    X_train_1, y_train_1 = preprocess(X_raw_1[:train_val_split], window_size, 0)
    X_val_1, y_val_1 = preprocess(X_raw_1[train_val_split:], window_size, 0)


    train_val_split = int(2/3*len(X_raw_2))
    X_train_2, y_train_2 = preprocess(X_raw_2[:train_val_split], window_size, 1)
    X_val_2, y_val_2 = preprocess(X_raw_2[train_val_split:], window_size, 1)

    print('Data Transformation:')
    print(X_train_1.shape)
    print(y_train_1.shape)
    print(X_train_2.shape)
    print(y_train_2.shape)
    print(X_val_1.shape)
    print(y_val_1.shape)
    print(X_val_2.shape)
    print(y_val_2.shape)

    X_train = np.vstack([X_train_1, X_train_2])
    y_train = np.hstack([y_train_1, y_train_2])
    X_val = np.vstack([X_val_1, X_val_2])
    y_val = np.hstack([y_val_1, y_val_2])

    print('Data Transformation:')
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    X_train, y_train = shuffle_data(X_train, y_train)
    X_val, y_val = shuffle_data(X_val, y_val)

    print('Data Transformation:')
    print(X_train.shape)
    print(X_val.shape)
    print(to_categorical(y_train).shape)
    print(to_categorical(y_val).shape)

    #import matplotlib.pyplot as plt
    #b_l,a_l = signal.butter(8, 0.01, 'low')
    b_h,a_h = signal.butter(8, 0.1, 'high')
    for i, sample in enumerate(X_train):
        for j in range(sample.shape[-1]):
            #temp = signal.filtfilt(b_l,a_l,sample[:,j])
            X_train[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])
    for i, sample in enumerate(X_val):
        for j in range(sample.shape[-1]):
            #temp = signal.filtfilt(b_l,a_l,sample[:,j])
            X_val[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])

    print('Data Transformation:')
    print(X_train.shape)
    print(X_val.shape)
    print(to_categorical(y_train).shape)
    print(to_categorical(y_val).shape)

    # define tensorflow model
    reg = 1e-3
    model = tf.keras.Sequential([ \
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
    '''
    Pure MLP:
    Flatten(input_shape=(window_size, 3)), \
    Dense(128, activation='relu', \
          kernel_regularizer=l2(reg), \
          bias_regularizer=l2(reg)), \
    Dense(64, activation='relu', \
          kernel_regularizer=l2(reg), \
          bias_regularizer=l2(reg)), \
    Dense(2, activation='softmax', \
          kernel_regularizer=l2(reg), \
          bias_regularizer=l2(reg)) \
    '''

    # define optimizer
    # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.save('model.h5')

    # train our model
    epochs = 10
    batch_size = 256
    history = model.fit(X_train, \
                    to_categorical(y_train), \
                    epochs=epochs, \
                    batch_size=batch_size,
                    validation_data = (X_val, to_categorical(y_val)))

    # Plot loss and accuracy
    plt.figure()
    plt.plot(range(epochs), history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()

    print('Sanity Check:')

    ###################################################################
    import collections
    print('========================================')
    X_test_1 = X_train_1
    X_test_2 = X_train_2
    for i, sample in enumerate(X_test_1):
        for j in range(sample.shape[-1]):
            X_test_1[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])
    for i, sample in enumerate(X_test_2):
        for j in range(sample.shape[-1]):
            X_test_2[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])

    results = model.predict(X_test_1, batch_size=256)

    classes = np.argmax(results, axis=1)
    temp = collections.Counter(classes)

    chance_0 = temp[0]/len(classes)
    chance_1 = temp[1]/len(classes)

    print('Xtrain_1')
    print('probability of class 0: {}'.format(chance_0))
    print('probability of class 1: {}'.format(chance_1))
    #print(y_train_1)

    results = model.predict(X_test_2, batch_size=256)

    classes = np.argmax(results, axis=1)
    temp = collections.Counter(classes)

    chance_0 = temp[0]/len(classes)
    chance_1 = temp[1]/len(classes)

    print('Xtrain_2')
    print('probability of class 0: {}'.format(chance_0))
    print('probability of class 1: {}'.format(chance_1))
    #print(y_train_2)
    ###################################################################
    print('========================================')
    X_test_1 = X_val_1
    X_test_2 = X_val_2
    for i, sample in enumerate(X_test_1):
        for j in range(sample.shape[-1]):
            X_test_1[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])
    for i, sample in enumerate(X_test_2):
        for j in range(sample.shape[-1]):
            X_test_2[i,:,j] = signal.filtfilt(b_h,a_h,sample[:,j])

    results = model.predict(X_test_1, batch_size=256)

    classes = np.argmax(results, axis=1)
    temp = collections.Counter(classes)

    chance_0 = temp[0]/len(classes)
    chance_1 = temp[1]/len(classes)

    print('Xval_1')
    print('probability of class 0: {}'.format(chance_0))
    print('probability of class 1: {}'.format(chance_1))
    #print(y_train_1)

    results = model.predict(X_test_2, batch_size=256)

    classes = np.argmax(results, axis=1)
    temp = collections.Counter(classes)

    chance_0 = temp[0]/len(classes)
    chance_1 = temp[1]/len(classes)

    print('Xval_2')
    print('probability of class 0: {}'.format(chance_0))
    print('probability of class 1: {}'.format(chance_1))
    #print(y_train_2)
