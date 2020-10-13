import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display

y, sr = librosa.load('test_input.wav', mono=True)
S = librosa.feature.melspectrogram(y, sr)
S_db_in = librosa.core.power_to_db(S)

print(y.shape)
print(S.shape)

y, sr = librosa.load('test_target.wav', mono=True)
S_db_tar = librosa.feature.melspectrogram(y, sr)
S_db_tar = librosa.core.power_to_db(S_db_tar)

plt.imshow(S_db_tar)
plt.show()