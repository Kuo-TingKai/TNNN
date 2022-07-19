from tensorflow.keras.datasets import fashion_mnist
#import matplotlib.pyplot as plt
import numpy
#from tensorflow.keras.utils import np_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D


# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()  # stored in ~/.keras/datasets/mnist.npz

# flatten 28*28 images to a 784 vector for each image
#num_pixels = X_train.shape[1] * X_train.shape[2]
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype( 'float32' )
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype( 'float32' )
# num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images
# X_train = X_train[:,:625]
# X_test = X_test[:,:625]


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train[:,:25,:25]
X_test = X_test[:,:25,:25]
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#print(num_classes)
#print(y_train)

# define model
# input 8 variables
# first hidden layer: 12 nodes; the second: 8 nodes; output layer: 1 node
model = Sequential()
model.add(Convolution2D(32, kernel_size=(5, 5), input_shape=(25, 25, 1), activation= 'relu' )) # 32 feature maps, 5x5 local receptive fields
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(128, activation= 'relu' ))
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
y_classes = [numpy.argmax(y, axis=None, out=None) for y in y_test]
for i in range(20):
        k = numpy.random.randint(0,len(y_test))
        print(preds[k], y_classes[k])
