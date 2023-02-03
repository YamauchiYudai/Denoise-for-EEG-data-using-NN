## ica.py
# Author: Yamauchi Yudai, 2022, RWTH Aachen Univ

## To make dataset for NN

import numpy as np

test_denoise_path = './nn_data/test_denoise.csv'
test_noise_path = './nn_data/test_noise.csv'
denoise_save_path = './nn_data/output_denoise.csv'
noise_save_path = './nn_data/output_noise.csv'

ica_denoise = np.loadtxt(test_denoise_path)
ica_noise = np.loadtxt(test_noise_path)
nn_denoise = np.loadtxt(denoise_save_path)
nn_noise = np.loadtxt(noise_save_path)

All_denoise = 0
All_noise = 0
best_denoise = 1
best_noise = 1
best = 0
js = 0
for i in range(ica_denoise.shape[0]):
    mse_denoise = ((ica_denoise[i,:] - nn_denoise[i,:])**2).mean(axis=0)
    mse_noise = ((ica_noise[i,:] - nn_noise[i,:])**2).mean(axis=0)
    All_denoise = All_denoise + mse_denoise
    All_noise = All_noise + mse_noise
    if best_noise > mse_noise:
        best_noise = mse_noise
        best = i
    if best_denoise > mse_denoise:
        best_denoise = mse_denoise
        best_noise = mse_noise
        js = i

print("best denoise",best_denoise,"j = ",js)
print("best noise",best_noise,"j = ",best)
print("denoise mse = ",(All_denoise/i),"noise mse = ",(All_noise/i))
