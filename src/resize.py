import cv2
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
def resize(x_train):
    x_train_resize = np.zeros((x_train.shape[0],25,25))
    for i in range(x_train.shape[0]):
        res = cv2.resize(x_train[i,:,:], dsize=(25, 25), interpolation=cv2.INTER_CUBIC)
        x_train_resize[i,:,:] = res
    return x_train_resize
x_train = resize(x_train)
x_test = resize(x_test)
x_train = np.save("../data/x_train.npy",x_train)
x_test = np.save("../data/x_test.npy",x_test)
y_train = np.save("../data/y_train.npy",y_train)
y_test = np.save("../data/y_test.npy",y_test)
