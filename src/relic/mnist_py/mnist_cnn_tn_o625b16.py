#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load mnist_cnn_tn.py
#import tensornetwork as tn
#import tensorflow as tf
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Flatten
##from tensornetwork.tn_keras.layers import DenseMPO
#from tensorflow.tn_keras.layers import Conv2DMPO
#from tensorflow.keras.layers import MaxPool2D
#import numpy as np

import tensornetwork as tn
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import  Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensornetwork.tn_keras.layers import DenseMPO
from tensornetwork.tn_keras.layers import Conv2DMPO
from tensorflow.keras.layers import MaxPool2D
import numpy as np


#import tensorflow.keras.utils
#from tensorflow.keras.utils import np_utils
#from tensorflow.keras.utils import to_categorical
#keras.utils

#from keras.datasets import mnist
##import matplotlib.pyplot as plt
#import numpy
#from keras.utils import np_utils
#
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import Flatten
#from keras.layers.convolutional import Convolution2D
#from keras.layers.convolutional import MaxPooling2D


# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # stored in ~/.keras/datasets/mnist.npz

# flatten 28*28 images to a 784 vector for each image
#num_pixels = X_train.shape[1] * X_train.shape[2]
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype( 'float32' )
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype( 'float32' )

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]


#print(num_classes)
#print(y_train)

# define model
model = Sequential()
model.add( Conv2DMPO(625, kernel_size=3, input_shape=(28, 28, 1), num_nodes=4, bond_dim=16, activation= 'relu' )) # 256 feature maps, 3x3 local receptive fields
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(num_classes, activation= 'softmax' ))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize layers
print(model.summary())

# fit the model on the dataset
# model.fit(X_train, y_train, epochs=10, batch_size=100)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)

# test
preds = model.predict_classes(X_test)
y_classes = [np.argmax(y, axis=None, out=None) for y in y_test]
for i in range(20):
        k = np.random.randint(0,len(y_test))
        print(preds[k], y_classes[k])


