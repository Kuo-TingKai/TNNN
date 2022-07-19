import numpy as np
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
np.save("train_X.npy",train_X)
np.save("train_y.npy",train_y)
np.save("test_X.npy",test_X)
np.save("test_y.npy",test_y)
