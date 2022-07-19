import tensornetwork as tn
import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
# from tensornetwork.tn_keras.layers import DenseMPO,  DenseDecomp, Conv2DMPO
# from tensorflow.keras.layers import MaxPool2D
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D
from tensornetwork.tn_keras.layers import DenseMPO,  DenseDecomp, Conv2DMPO
from keras.layers import MaxPool2D
import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import time

def init_model(model_arch,num_nodes, bond_dim):

    """
    types of model architecture:

      1. mlp
      2. mpo      (*)
      3. decomp   (*)
      4. cnn
      5. cnnmpo   (*)
      6. cnndecomp(*)
      7. lenet5
      8. tnlenet5 (*)
      9. vgg16
      10.tnvgg16  (*)

    """

    num_classes = 10

    if model_arch=='mlp':
        MLP_model = Sequential([
          Flatten(input_shape=(25, 25)),
          #tf.keras.layers.Dense(128, activation='relu'),
          Dense(256, use_bias=True, activation='relu'),
          Dropout(0.5),
          Dense(num_classes, activation= 'softmax' )
        ])
        ##print("MLP:")
        ##print(MLP_model.summary())
        return MLP_model

    elif model_arch=='mpo':
        MPO_model = Sequential([
          Flatten(input_shape=(25, 25)),
          #tf.keras.layers.Dense(128, activation='relu'),
          DenseMPO(256, num_nodes=num_nodes, bond_dim=bond_dim, use_bias=True, activation='relu'),
          Dropout(0.5),
          Dense(num_classes, activation= 'softmax' )
        ])
        # #print("MPO:")
        # #print(MPO_model.summary())
        return MPO_model

    elif model_arch=='decomp':
        Decomp_model = Sequential([
          Flatten(input_shape=(25, 25)),
          DenseDecomp(256, decomp_size=128, use_bias=True, activation='relu'),
          Dropout(0.5),
          Dense(num_classes,  use_bias=True, activation= 'softmax' )
        ])
        # #print("Decomp:")
        # #print(Decomp_model.summary())
        return Decomp_model

    elif model_arch=='cnn':
        CNN_model = Sequential([
          Conv2D(64, kernel_size=3, input_shape=(25, 25, 1), use_bias=True, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Dropout(0.5),
          Flatten(),
          Dense(num_classes,  use_bias=True, activation= 'softmax' )
        ])
        # #print("CNN:")
        # #print(CNN_model.summary())
        return CNN_model

    elif model_arch=='cnnmpo':
        # not clear....
        CNNMPO_model = Sequential([
          Conv2DMPO(64, kernel_size=5, num_nodes=num_nodes, bond_dim=bond_dim, input_shape=(25, 25, 1), use_bias=True, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Dropout(0.5),
          Flatten(),
          Dense(num_classes,  use_bias=True, activation= 'softmax' )
        ])
        # #print("CNNMPO:")
        # #print(CNNMPO_model.summary())
        return CNNMPO_model
    
    elif model_arch=='cnndecomp':        
        CNN_Decomp_model = Sequential([
          Conv2D(64, kernel_size=3, input_shape=(25, 25, 1), use_bias=True, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Dropout(0.5),
          Flatten(),
          DenseDecomp(num_classes, decomp_size=128, use_bias=True, activation='relu'),
          #Dropout(0.5),
          Dense(num_classes,  use_bias=True, activation= 'softmax' )
        ])
        # #print("CNN_Decomp:")
        # #print(CNN_Decomp_model.summary())
        return CNN_Decomp_model

    elif model_arch=='lenet5':
        LeNet5_model = Sequential([
          Conv2D(6, kernel_size=(5, 5), input_shape=(25, 25, 3), padding='same', use_bias=False, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(16, kernel_size=(5, 5), use_bias=False, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          #Dropout(0.5),
          Flatten(),
          Dense(120, use_bias=False, activation='relu'),
          Dense(84, use_bias=False, activation='relu'),
          Dense(num_classes,  use_bias=False, activation= 'softmax' )
        ])
        #print("LeNet5:")
        #print(LeNet5_model.summary())
        return LeNet5_model

    elif model_arch=='tnlenet5': 
        tn_LeNet5_model = Sequential([
          Conv2D(6, kernel_size=(5, 5), input_shape=(25, 25, 3), use_bias=False, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(16, kernel_size=(5, 5), use_bias=False, activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2), strides=1),
          #Dropout(0.5),
          Flatten(),
          #Dense(120, use_bias=True, activation='relu'),
          #Dense(84, use_bias=True, activation='relu'),
          DenseMPO(256, num_nodes=num_nodes,bond_dim=bond_dim, use_bias=False, activation='relu'),  # input 400
          #DenseDecomp(81, decomp_size=128, use_bias=True, activation='relu'),
          DenseMPO(81, num_nodes=num_nodes, bond_dim=bond_dim, use_bias=False, activation='relu'),
          Dense(num_classes,  use_bias=False, activation= 'softmax' )
        ])
        #print("tn_LeNet5:")
        #print(tn_LeNet5_model.summary())
        return tn_LeNet5_model

    elif model_arch=='vgg16':
        VGG16_model = Sequential([
          Conv2D(64, kernel_size=(3, 3), input_shape=(25, 25, 3), padding='same', use_bias=False, activation= 'relu'),
          Conv2D(64, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu'),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(128, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(128, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(256, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(256, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(256, kernel_size=(1, 1), use_bias=False, padding='same', activation= 'relu' ), 
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(1, 1), use_bias=False, padding='same', activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(1, 1), use_bias=False, padding='same', activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          #Dropout(0.5),
          Flatten(),
          Dense(4096, use_bias=False, activation='relu'),
          Dense(4096, use_bias=False, activation='relu'),
          #Dense(1000, use_bias=False, activation='relu'),
          Dense(num_classes, use_bias=False, activation= 'softmax' )
        ])
        #print("VGG16:")
        #print(VGG16_model.summary())
        return VGG16_model

    elif model_arch=='tnvgg16':
        tn_VGG16_model = Sequential([
          Conv2D(64, kernel_size=(3, 3), input_shape=(25, 25, 3), padding='same', use_bias=False, activation= 'relu'),
          Conv2D(64, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu'),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(128, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(128, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(256, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(256, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(256, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(256, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2D(512, kernel_size=(3, 3), use_bias=False, padding='same', activation= 'relu' ),
          Conv2DMPO(512, kernel_size=3, num_nodes=num_nodes, bond_dim=bond_dim, padding='same', use_bias=False, activation= 'relu'),
          Conv2DMPO(512, kernel_size=3, num_nodes=num_nodes, bond_dim=bond_dim, padding='same', use_bias=False, activation= 'relu'),
          MaxPool2D(pool_size=(2, 2)),
          Flatten(),
          #DenseMPO(4096, num_nodes=3, bond_dim=8, use_bias=False, activation='relu'),  # input 512
          #DenseMPO(4096, num_nodes=3, bond_dim=8, use_bias=False, activation='relu'),
          Dense(4096, use_bias=False, activation='relu'),
          Dense(4096, use_bias=False, activation='relu'),
          Dense(num_classes, use_bias=False, activation= 'softmax' )
        ])
        #print("tn_VGG16:")
        #print(tn_VGG16_model.summary())
        return tn_VGG16_model