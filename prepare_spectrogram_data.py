import numpy as np
from numpy import save
import librosa
import os

train_dir = '/Volumes/DatasetHD/musdb18hq/train/'
train_file_dirs = os.listdir(train_dir)
train_file_dirs.remove('.DS_Store')

SL = 110080  # sample length - 512 (hop_length) * 215 (frames / spectrogram)
n_fft = 2048  # important for Fourier Transform wish I understood it better
hop_length = 512 # number of samples per frame in spectrograms

X = []
Y = []
count = 0
for d in train_file_dirs:
    count += 1
    print(count, '/ 100 :', d)

    wav_files = os.listdir(train_dir + d)
    
    mix, sr = librosa.load(train_dir + d + '/mixture.wav')
    vocal, sr = librosa.load(train_dir + d + '/vocals.wav')

    trail = np.zeros(SL - mix.shape[0] % SL)
    mix = np.hstack((mix, trail))
    vocal = np.hstack((vocal, trail))

    for i in range(int(mix.shape[0] / SL)):
        mix_snip = mix[SL * i : SL * i + SL]
        vocal_snip = vocal[SL * i : SL * i + SL]

        mix_spec = librosa.feature.melspectrogram(mix_snip, sr, n_fft=n_fft, hop_length=hop_length)
        mix_spec = librosa.core.power_to_db(mix_spec)

        vocal_spec = librosa.feature.melspectrogram(vocal_snip, sr, n_fft=n_fft, hop_length=hop_length)
        vocal_spec = librosa.core.power_to_db(vocal_spec)

        X.append(mix_spec)
        Y.append(vocal_spec)

X = np.array(X)
Y = np.array(Y)

ixs = np.array(range(X.shape[0]))
np.random.shuffle(ixs)

X = X[ixs]
Y = Y[ixs]

print('\nSaving Data...')
save('musdb_spec_inputs.npy', X)
save('musdb_spec_targets.npy', Y)
print('Done')