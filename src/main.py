import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# INFO not be printed
import tensornetwork as tn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from tensornetwork.tn_keras.layers import DenseMPO,  DenseDecomp, Conv2DMPO
import numpy as np
# import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import matplotlib.pyplot as plt
# import sys
# import time

import argparse

from util.save_history import *
from  util.timestamp import *
from util.save_load_model import *

from init_model import *

parser = argparse.ArgumentParser()
parser.add_argument("ac",type=str,help="model architecture",default="mpo")
parser.add_argument("ep", type=int,help="bond dimension",default=1)
parser.add_argument("bd", type=int,help="bond dimension",default=4)
parser.add_argument("n",type=int,help="number of nodes",default=4)
parser.add_argument("d",type=str,help="dataset",default="mnist")
args = parser.parse_args()

bond_dim = args.bd
num_nodes = args.n
model_arch = args.ac
dataset = args.d
nepoch = args.ep
print(model_arch,nepoch,bond_dim,num_nodes,dataset)
num_classes = 10


print("load dataset...")
x_train = np.load("./data/"+dataset+"/resize/x_train.npy")
y_train = np.load("./data/"+dataset+"/resize/y_train.npy")
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
print("loaded")

# # 625 = 25x25 = 5x5x5x5
# # 256 = 16x16 = 4x4x4x4
# model = Sequential([
#   Flatten(input_shape=(25, 25)),
#   #tf.keras.layers.Dense(128, activation='relu'),
#   DenseMPO(256, num_nodes=num_nodes, bond_dim=bond_dim, use_bias=True, activation='relu'),
#   Dropout(0.5),
#   # Dense(num_classes, activation= 'softmax' )
#   Dense(num_classes,activation='softmax')
# ])
# #print("MPO:")
# #print(model.summary())


model = init_model(model_arch,num_nodes,bond_dim)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("training model...")
history = model.fit(x_train, y_train, validation_split=0.2, epochs=nepoch, batch_size=16, verbose=1)

print("model is trained, save the history...")
save_history(history,model_arch,nepoch,bond_dim,num_nodes,dataset)
model_dir = creat_model_dir(model,model_arch,nepoch,bond_dim,num_nodes,dataset)
print(model_dir)
model.save(model_dir)
print("history is saved.")