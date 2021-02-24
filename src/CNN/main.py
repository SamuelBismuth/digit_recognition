'''
Python version: 3.8
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://missinglink.ai/guides/convolutional-neural-networks/python-convolutional-neural-network-creating-cnn-keras-tensorflow-plain-python/


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import repackage
repackage.up()
from data import split_data

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000,28,28,1)
# X_test = X_test.reshape(10000,28,28,1)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_train[0]
# print(type(X_train))

def reshape_data(train_x, test_x, train_y, test_y):
    train_x=np.asarray(train_x.tolist()).astype(np.float32).reshape(29400,28,28,1)
    test_x=np.asarray(test_x.tolist()).astype(np.float32).reshape(12600,28,28,1)
    train_y=np.asarray(train_y.tolist()).astype(np.float32)
    test_y=np.asarray(test_y.tolist()).astype(np.float32)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, test_x, train_y, test_y

def CNN(X_train,X_test,y_train,y_test):
    X_train,X_test,y_train,y_test = reshape_data(X_train,X_test,y_train,y_test)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Loss Function')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    CNN(train_x, test_x, train_y, test_y)