import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold,train_test_split

data_path = 'train_set'
train_data_path = './nn_data/train_data.csv'
train_denoise_path = './nn_data/train_denoise.csv'
train_noise_path = './nn_data/train_noise.csv'
test_data_path = './nn_data/test_data.csv'
test_denoise_path = './nn_data/test_denoise.csv'
test_noise_path = './nn_data/test_noise.csv'
dataset_path = os.path.join(data_path,'train.csv')
denoise_path = os.path.join(data_path,'ica_result.csv')
noise_path = os.path.join(data_path,'ica_noise.csv')
x = np.loadtxt(dataset_path)
y = np.loadtxt(denoise_path)
z = np.loadtxt(noise_path)

x_train, x_test = train_test_split(x, train_size=0.9)
y_train, y_test = train_test_split(y, train_size=0.9)
z_train, z_test = train_test_split(z, train_size=0.9)
np.savetxt(train_data_path,x_train)
np.savetxt(train_denoise_path,y_train)
np.savetxt(train_noise_path,z_train)
np.savetxt(test_data_path,x_test)
np.savetxt(test_denoise_path,y_test)
np.savetxt(test_noise_path,z_test)