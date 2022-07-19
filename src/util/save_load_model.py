import keras

import os
from  util.timestamp import time_stamp
import numpy as np

# bond_dim = args.bd
# num_nodes = args.n
# model_arch = args.ac
# dataset = args.d
# nepoch = args.ep


def creat_model_dir(model,model_arch,nepoch,bond_dim,num_nodes,dataset):
    time = time_stamp()    
    model_path ="./model_weight/"+model_arch
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    model_path+="/"+time
    model_path+="/"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)   
    args = str(nepoch)+"_"+str(bond_dim)+"_"+str(num_nodes)+"_"+str(dataset)
    model_path+=args
    if not os.path.isdir(model_path):
        os.mkdir(model_path)  

    return model_path+"/model.h5"

    # if not os.path.isdir("./model_weight/"+model_arch):
    #     os.mkdir("./model_weight/"+model_arch)
    #     if not os.path.isdir("./model_weight/"+model_arch+"/"+time):
    #         os.mkdir("./model_weight/"+model_arch+"/"+time)
    #         model_path = "./model_weight/"+model_arch+"/"+time+"/"
    #         args = str(nepoch)+"_"+str(bond_dim)+"_"+str(num_nodes)+"_"+str(dataset) 
    #         model_path += args
    #         if not os.path.isdir(model_path):
    #             os.mkdir(model_path)
    #             return model_path+'/model.h5'

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model
