import numpy as np
import os
path_dir = "./data"
temp = False
data = None
for file in os.listdir(path_dir):
    print(file)
    _data = np.load(path_dir+"/"+file)
    if(temp):
        data = np.concatenate((data, _data), axis = 0)
    else:
        data = _data
        temp = True

print(data.shape)
np.save("./dataset/data.npy", data)