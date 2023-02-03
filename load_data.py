import numpy as np

clean = np.load('general_data/clean_EEG_test.npy')
train = np.load('general_data/noiseEEG_test.npy')
np.savetxt('train_set/truth.csv',clean)
np.savetxt('train_set/train.csv',train)