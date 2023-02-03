import mne 
import pandas as pd
import os
import numpy as np

components = 10
ch = 20
csv_path = r'train_set/train.csv'
save_path = r"train_set/ica_result.csv"
noise_path = r"train_set/ica_noise.csv"
correct_path = r"train_set/truth.csv"

noise_data  = pd.read_table(csv_path, sep="\s+", header=None)
noise_data = noise_data.values # DataFrame to ndarry
correct_data  = pd.read_table(correct_path, sep="\s+", header=None)
correct_data = correct_data.values # DataFrame to ndarry
info = mne.create_info(ch_names=ch,sfreq=256,ch_types="eeg")
print(noise_data.shape)
all_results = np.zeros(noise_data.shape)
all_noise = np.zeros(noise_data.shape)
for i in range(0,3400,ch):#適切なチャネル数でないとICAはうまくいかないhttps://www.jstage.jst.go.jp/article/sicejl/50/6/50_418/_pdf
  eeg = noise_data[i:i+ch,:]
  raw  = mne.io.RawArray(eeg, info)
  org_raw = raw.copy()
  noise_include = raw.copy()

  method = 'infomax'
  random_state = 1
  ica=mne.preprocessing.ICA(n_components=components,method=method,random_state=random_state)

  ica.fit(raw)
  ica.fit(noise_include)

  # Retrieve explained variance
  # unitize variances explained by PCA components, so the values sum to 1
  pca_explained_variances = ica.pca_explained_variance_ / ica.pca_explained_variance_.sum()

  # Now extract the variances for those components that were used to perform ICA
  ica_explained_variances = pca_explained_variances[:ica.n_components_]

  for idx, var in enumerate(ica_explained_variances):
      print(
          f'Explained variance for ICA component {idx}: '
          f'{round(100 * var, 1)}%'
      )
  best_error = 0.000001
  bad_error = 1000
  error = np.zeros(components)
  best_raw = np.zeros((ch,512))
  best_noise = np.zeros((ch,512))
  for snumber in range (components):
    ica.apply(raw,include = [snumber] )
    raw.load_data()
    ndarray_raw = raw.get_data()
    all_mse = 0 
    mse = np.mean((correct_data[i:i+ch,:] - ndarray_raw) ** 2)
    error[snumber] = mse
    if mse > best_error:
      best_raw = ndarray_raw
      best_error = mse
    if mse < bad_error:
      best_noise = ndarray_raw
      bad_error = mse
    all_results[i:i+ch,:] = best_raw
    all_noise[i:i+ch,:] = best_noise

all_mse = np.mean((correct_data - all_results) ** 2)
print("total = ",all_mse)

np.savetxt(save_path,all_results)
np.savetxt(noise_path,all_noise)