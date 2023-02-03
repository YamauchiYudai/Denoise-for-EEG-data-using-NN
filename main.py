import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold,train_test_split
from nn import Net


PATH = './model/denoise_model.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = 'train_set'
train_data_path = './nn_data/train_data.csv'
train_denoise_path = './nn_data/train_denoise.csv'
train_noise_path = './nn_data/train_noise.csv'
test_data_path = './nn_data/test_data.csv'
test_denoise_path = './nn_data/test_denoise.csv'
test_noise_path = './nn_data/test_noise.csv'
denoise_save_path = './nn_data/output_denoise.csv'
noise_save_path = './nn_data/output_noise.csv'
x_train = np.loadtxt(train_data_path)
x_test = np.loadtxt(test_data_path)
y_train = np.loadtxt(train_denoise_path)
y_test = np.loadtxt(test_denoise_path)
z_train = np.loadtxt(train_noise_path)
z_test = np.loadtxt(test_noise_path)

x = Variable(torch.from_numpy(x_train).float(), requires_grad=True)
y = Variable(torch.from_numpy(y_train).float())
z = Variable(torch.from_numpy(z_train).float())
x_test = Variable(torch.from_numpy(x_test).float(), requires_grad=True)
y_test = Variable(torch.from_numpy(y_test).float())
z_test = Variable(torch.from_numpy(z_test).float())

denoise_result = np.zeros((y_test.shape))
noise_result = np.zeros((z_test.shape))
model = Net().to(device)
optimizer = optim.Adam(model.parameters(),lr = 1e-3)
criterion = nn.MSELoss()
kf = KFold(n_splits=9, shuffle=True, random_state=1)
valid_loss = 0
min_loss = 1000
cv = 0

def train():
  for i in range(1000):
    epoch_loss = 0.0
    step = 0
    for j in range(train_data.shape[0]):
      # zero the parameter gradients
      step += 1
      optimizer.zero_grad()
      output = model((train_data[j,:].reshape(512)).T.float().to(device))
      correct_data = np.concatenate([train_denoise[j,:],train_noise[j,:]],0).reshape(1024)
      tensor_correct =Variable(torch.from_numpy(correct_data).float())
      train_loss = criterion(output,tensor_correct.to(device))
      train_loss.backward(retain_graph=True)
      optimizer.step()
      epoch_loss += train_loss.item()
    print("epoch %d loss:%0.3f" % (i+1, epoch_loss/step))

def val():
   with torch.no_grad():
     global  min_loss
     all_val_loss = 0
     step = 0
     for j in range(valid_data.shape[0]):
      # zero the parameter gradients
      step += 1
      optimizer.zero_grad()
      output = model((valid_data[j,:].reshape(512)).T.float().to(device))
      correct_data = np.concatenate([valid_denoise[j,:],valid_noise[j,:]],0).reshape(1024,1)
      valid_correct =Variable(torch.from_numpy(correct_data).float())
      valid_loss = criterion(output, valid_correct[j,:].to(device))
      all_val_loss += valid_loss.item()
     print("all_val_loss:%0.3f" %(all_val_loss/step))
     if (all_val_loss/step) < min_loss:
      min_loss = (all_val_loss/step)
      best_model = model
      torch.save(best_model.state_dict(), PATH)

# test
def test():
 with torch.no_grad():
  model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
  for j in range(x_test.shape[0]):
    outputs = model((x_test[j,:].reshape(512)).T.float().to(device))
    npout = (outputs.to('cpu').detach().numpy().copy()).T
    denoise,noise = np.split(npout,2)
    denoise_result[j,:] = denoise
    noise_result[j,:] = noise
  np.savetxt(denoise_save_path,denoise_result)
  np.savetxt(noise_save_path,noise_result)

if __name__ == '__main__':
  cv += valid_loss / kf.n_splits
  number = 1
  print("Start to Traning")
  for _fold, (train_index, valid_index) in enumerate(kf.split(x)):
    number +=1
    train_data,valid_data = x[train_index], x[valid_index]
    train_denoise, valid_denoise = y[train_index], y[valid_index]
    train_noise, valid_noise = z[train_index], z[valid_index]
    train()
    print("Finish to Traning，Start to Validation")
    val()
    print("Finish to Validation，Start to %d training" %(number))
  print("Finish to All_train，Start to Test")
  test()
  print("Finish to Test，ALL DONE")
