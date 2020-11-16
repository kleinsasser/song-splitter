# Vocal Isolation from Music Using Fourier Transforms and Deep Convolutional Neural Networks in Tensorflow
Max Kleinsasser

## Summary
This repository contains a convolutional u-net model based on [this paper](https://arxiv.org/pdf/1903.01415.pdf) from 2017. The model takes the magnitude spectrograms of songs containing a mixture of vocals and accompaniment and returns a spectrogram of the same song stripped of its accompaniment, leaving only the vocals. I built this model as part of a larger personal project of mine and want to share my experience to perhaps aid others in their own implementation.

## Dataset
The data this model was trained on is from the [musdb18hq dataset](https://sigsep.github.io/datasets/musdb.html). The paper whose methodology is used in this project uses a separate dataset from files (about 20,000 song stems) mined from the commercial sources. I was able to acheive acceptable results from the 100 song stems from musdb. Use the link to request access to the dataset.

## Data Pre-processing
The model used in this project is trained on magnitude spectrograms of .wav files containing song mixtures (vocals + instruments) and vocal-only. To achieve this I split the raw files into ~11.8 second patches and used the short-time fourier transform (stft) function from the python package [librosa](https://librosa.org) to aquire the magnitude spectrograms along with the phase matrices (for later signal reconstruction). The stft function computes a complex-valued matrix in the time-frequency domain corresponding to the input signal, the magnitude spectrogram is computed by taking the absolute value of this matrix. I used the same stft configurations as the paper. The shape of the inputs and outputs of the training set after pre-processing were (1963, 512, 128, 1) -> (n_samples, n_frequency_bins, n_time_steps, n_image_channels).

## The Model
The model is a convolutional neural network based on the u-net architecture as explained in the reference paper. It has 6 downsampling convolutional layers and 6 upsampling deconvolutional layers. The network uses batch normalization on every layer and a ReLU activation (except for the last upsampling layer, which has a sigmoid activation). Each upsampling layer is also concatenated with its shape-corresponding downsampling layer. The goal of the network is to learn a mask that when multiplied by the input spectrogram, yeilds the spectrogram of the vocal-only or target spectrogram. The paper uses the L1 norm as a loss function, I used the extremely similar loss: mean absolute error. The model is trained with the Adam optimizer with parameter 0.001.

I built this model using a tf.keras.Model, which has all of the benefits of a keras Sequential model (built-in losses, optimizers, and training loops) while allowing for the non-sequential skip connections present in the u-net architecture. I trained the model on my Macbook Pro, which has a 3.5 GHz Dual-Core Intel i7 processor and 16GB RAM for 30 epochs, training took ~1.5 hours. For reference, the final MAE settled around (training: 0.1, validation: 0.115, testing: 0.125), but given the subjective nature of the task results are best measured with ears.

(See [model_graph.png](model_graph.png) for the model graph, too long to include here)

## Signal Reconstruction
The model is trained exclusively on magnitude spectrogram data, reconstructing a signal from a stft requires both the magnitude information and the phase information. Given the predicted magnitude spectrogram M and the phase information matrix of the input P, the final signal is reconstructed using librosa.istft(M * P). Using the phase information from the input to construct the output seems to work fairly well qualitatively.

## Some Results

Some examples of the model's performance:

