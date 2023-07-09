import numpy as np
import os
# Trước khi chạy kéo file A.npy ra khỏi folder "data" và cùng cấp với nó
path_1 = "./A.npy"
data = np.load(path_1)
path_dir = "./data"
for file in os.listdir(path_dir):
    A = np.load(path_dir+"/"+file)
    data = np.concatenate((data, A), axis = 0)

print(data.shape)
np.save("./dataset/data.npy", data)