# Denoise-for-EEG-data-using-NN

## Introduction
Electroencephalogram (EEG) is one of the methods to measure brain function. EEG
records the electrical activity generated by the brain using a probe placed on the scalp to measure brain function. However, the EEG signal contains a lot of noise, so it needs to be denoised. One technique to remove noise from the signal is independent component analysis (ICA). ICA assume that components are statistically independent of each other and separate mixed components and ICA is often used for denoise in EEG. However ICA has many disadvantages. For example, ICA has a number of components and is not real-time. Therefore we assume to use the Neural Network (NN). NN is one of the Machine Learning technics. NN can learn algorithms by learning to produce desired outputs in response to input data. In this project trained NN by noise and denoise EEG data created by ICA and verify the accuracy of the NN output compared to ICA results.

You can download the dataset I used from (https://github.com/ncclabsustech/Single-Channel-EEG-Denoise)


#### References
1. Stropahl, Maren, et al. "Source-modeling auditory processes of EEG data using EEGLAB and brainstorm." Frontiers in neuroscience 12 (2018): 309.
2. Zhang, Haoming, et al. "EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising." arXiv preprint arXiv:2009.11662 (2020).
3. Karhunen, J., Oja, E., Wang, L., Vigario, R., & Joutsensalo, J. (1997). A class of neural networks for independent component analysis. IEEE Transactions on neural networks, 8(3), 486-504.


## User manual
###File contents
    ・ica.py
    ・make_dataset.py
    ・main.py
    ・nn.py
    ・caculate_test_mse.py
    ・load_data.py
    model
        ・denoise_model.pt
    nn_data
    train_set
    general_data
        ・clean_EEG_test.npy
        ・noiseEEG_test.npy
 ## User manual

### File contents

- `・ica.py`
- `・make_dataset.py`
- `・main.py`
- `・nn.py`
- `・caculate_test_mse.py`
- `・load_data.py`
- `model`
    - `・denoise_model.pt`
- `nn_data`
- `train_set`
- `general_data`
     - `・clean_EEG_test.npy`
     - `・noiseEEG_test.npy`

 
##Flow Chart
1. load_data.py
This program creates a dataset from the web(https://github.com/ncclabsustech/Single-Channel-EEG-Denoise)
. In this time, you define that train.csv is
Single-Channel-EEG-Denoise/Example_data/EOG/1/ noiseEEG_test.npy, truth.csv is
Single-Channel-EEG-Denoise/Example_data/EOG/1/ cleanEEG_test.npy.
2. ica.py
You must download mne[1]
, before excusing this program. mne is an Open-source Python
package for exploring, visualizing, and analyzing human neurophysiological data. This
program separates noise data and denoise data. You excuse this program, ica_result.csv
and ica_noise.csv are created by train_set.
3. make_dataset.py
This program makes the dataset for training NN. You excuse this program, train_data.csv
and train_denoise.csv, train_noise.csv, test_data.csv, test_denoise.csv, test_noise.csv are
created by nn_data.
4. main.py
This program creates noise and denoise data using a neural network. This program
usesng PyTorch and numpy , pandas, sklearn. You must download these before excusing
them. You excuse this program, ica_result.csv and ica_noise.csv are created output_noise
and output_denoise.
5. caculate_test_mse.py
This program calculates for results of training which compares of results of ICA.

###Other programs
1. nn.py
This program writes a model of neural network for main.py.
2 denoise_model.pt
This is best model of NN
