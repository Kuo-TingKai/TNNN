import tensornetwork as tn
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
#from tensornetwork.tn_keras.layers import DenseMPO
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D, AveragePooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensornetwork.tn_keras.layers import DenseMPO,  DenseDecomp, Conv2DMPO
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


arguments = len(sys.argv) - 1
if arguments >= 3:
        E = int(sys.argv[1])
        B = int(sys.argv[2])
        D = int(sys.argv[3])
else:
        E = 10
        B = 100
        D = 8

o_file = 'history_fashion_tn_LeNet5_E{0}B{1}D{2}.pdf'.format(E, B, D)

# fix random seed for reproducibility
seed = 5
np.random.seed(seed)

# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()  # stored in ~/.keras/datasets/mnist.npz

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



# define model

model = Sequential([
  Conv2D(6, kernel_size=(5, 5), input_shape=(28, 28, 1),  padding='same', use_bias=False, activation= 'relu' ), 
  BatchNormalization(),
  Dropout(0.3),
  MaxPool2D(pool_size=(2, 2)),
  Conv2D(16, kernel_size=(5, 5),  padding='valid', use_bias=False, activation= 'relu' ),
  BatchNormalization(),
  Dropout(0.4),
  MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
  #Conv2D(120, kernel_size=(5, 5),  padding='valid', use_bias=False, activation= 'relu' ),
  #Conv2DMPO(256, kernel_size=5, num_nodes=4, bond_dim=D, input_shape=(28, 28, 1), use_bias=True, activation= 'relu' ),
  #BatchNormalization(),
  #Dropout(0.4),

  Flatten(),
  #Dense(81, use_bias=False, activation='relu'),
  DenseMPO(256, num_nodes=4, bond_dim=D, use_bias=True, activation='relu'),  # input 1296
  BatchNormalization(),
  Dropout(0.5),
  DenseMPO(81, num_nodes=4, bond_dim=D, use_bias=False, activation='relu'),
  BatchNormalization(),
  Dense(num_classes,  use_bias=False, activation= 'softmax' )
])


# compile the model
tf.keras.regularizers.l2(0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize layers
print(model.summary())

# fit the model on the dataset
# model.fit(X_train, y_train, epochs=10, batch_size=100)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=E, batch_size=B)
#############################
# plot history
###########################
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
labels = ["train", "test"]

l1=ax1.plot(history.history['accuracy'], color='green', label = 'train')
l2=ax1.plot(history.history['val_accuracy'], color='red', label = 'test')
ax1.set_ylabel('accuracy', fontsize=14)
ax1.set_ylim([0, 1.02])

ax2.plot(history.history['loss'], color='green', label = 'train')
ax2.plot(history.history['val_loss'], color='red', label = 'test')
ax2.set_ylabel('loss', fontsize=14)
ax2.set_xlabel('epoch', fontsize=14)
ax2.set_ylim([0, 1.02])

legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')

plt.savefig(o_file)
plt.show()

#############################
# test
#############################
preds = np.argmax(model.predict(X_test), axis=-1)
y_classes = [np.argmax(y, axis=None, out=None) for y in y_test]
preds_f=[]
count = 0
for k in range(len(y_test)):
        #k = np.random.randint(0,len(y_test))
        if preds[k] !=  y_classes[k]:
                #print(preds[k], y_classes[k])
                #if k not in preds_f:
                preds_f.append(k)
                count = count + 1
print("Error rate = ", count/len(y_test))
# plot
cols = 5
if len(preds_f) < 5:
        cols = len(preds_f)
rows = int(len(preds_f) / 5)
if len(preds_f) % 5 != 0:
        rows = rows +1
if rows > 2:
        rows = 2
axes=[]
fig=plt.figure()
for i in range(cols*rows):
        #k = preds_f[i]
        j = np.random.randint(0,len(preds_f))
        k = preds_f[j]
        axes.append( fig.add_subplot(rows, cols, i+1) )
        subplot_title=(str(preds[k])+" ("+str(y_classes[k])+")")
        axes[-1].set_title(subplot_title)
        axes[-1].set_axis_off()
        plt.imshow(X_test[k])
fig.tight_layout()
plt.savefig('plot.pdf')
plt.show()

