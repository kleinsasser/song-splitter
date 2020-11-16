'''
This file splits the training data of musdb18hq to 10 second snippets
of each mixture.txt and vocals.txt file and saves the snippets to .npy
files of the processed .wav files
'''
import numpy as np
from numpy import savetxt
import librosa
import os

train_dir = '/musdb18hq/train/'
train_file_dirs = os.listdir(train_dir)
train_file_dirs.remove('.DS_Store')

SL = 22050 # 1 second

X = []
Y = []
count = 0
for d in train_file_dirs:
    print(count, '/ 100 Processing:', d)
    count += 1

    wav_files = os.listdir(train_dir + d)
    
    mix, _ = librosa.load(train_dir + d + '/mixture.wav')
    vocal, _ = librosa.load(train_dir + d + '/vocals.wav')

    trail = np.zeros(SL - mix.shape[0] % SL)
    mix = np.hstack((mix, trail))
    vocal = np.hstack((vocal, trail))

    for i in range(int(mix.shape[0] / SL)):
        X.append(mix[SL * i : SL * i + SL])
        Y.append(vocal[SL * i : SL * i + SL])

    if count == 10: break

X = np.array(X)
Y = np.array(Y)

np.save('1sec_inputs.csv', X)
np.save('1sec_targets.csv', Y)