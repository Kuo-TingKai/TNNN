import os
from  util.timestamp import time_stamp
import numpy as np
def save_history(history,model_arch,nepoch,bond_dim,num_nodes,dataset):
    time = time_stamp()
    save_path = "./history/"+model_arch+"/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        save_path+=time
        if not os.path.isdir():
            os.mkdir(save_path)
            args = str(nepoch)+"_"+str(bond_dim)+"_"+str(num_nodes)+"_"+str(dataset)
            save_path+="/"
            save_path+=args
            save_path+="/"
            if not os.path.isdir():
                os.mkdir(save_path)
    #    os.mkdir("./history/"+model_arch)

        # for k,v in history.history.items():
        #     file = "./history/"+model_arch+"/"+k+"_"+time_stamp()+".npy"
        #     #if not os.path.exists(file):
        #     np.save(file,np.array([]))

    for k,v in history.history.items():
        
        #date = int(time[3:5])
        #hour = int(time[-5:-3])
        #minute = int(time[-2:])
        
        #if time[-2:]-prev_time[-2]
        
        np.save(save_path+k+"_"+".npy",v)
        
        #pdate = date
        #phour = hour
        #pminute = minute