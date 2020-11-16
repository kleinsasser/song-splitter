def show_spectrograms(S_input, S_target, S_pred):
    
    D1 = librosa.amplitude_to_db(S_input)
    D2 = librosa.amplitude_to_db(S_target)
    D3 = librosa.amplitude_to_db(S_pred)
    
    plt.figure()
    ax1 = plt.subplot(3,1,1)
    librosa.display.specshow(D1)
    plt.title('Input')
    plt.colorbar(format='%+2.0f dB')

    ax2 = plt.subplot(3,1,2)
    librosa.display.specshow(D2)
    plt.title('Target')
    plt.colorbar(format='%+2.0f dB')
    
    ax3 = plt.subplot(3,1,3)
    librosa.display.specshow(D3)
    plt.title('Prediction')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

def spectrogram_to_wavfile(filename, S):
    S = librosa.core.db_to_power(S)
    print('Inverting spectrogram...')
    y = librosa.feature.inverse.mel_to_audio(S)
    print('Creating {}...'.format(filename))
    scipy.io.wavfile.write(filename, 22050, y)
    print('Done')