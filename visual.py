import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

fig_size = (12,10)
file = "train/00001.wav"
sig, sr = librosa.load(file, sr=16000) #data fetch

#Waveform
def Waveform(sig):
    plt.figure(figsize=fig_size)
    librosa.display.waveshow(sig, sr, alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()

#Fourier Transform; FT -> spectrum
def fft(sig):
    fft = np.fft.fft(sig)
# take absolute value; magnitude
    magnitude = np.abs(fft)
# Frequency
    f = np.linspace(0,sr,len(magnitude))
# specturm is symetric; using half
    left_spectrum = magnitude[:int(len(magnitude)/2)]
    left_f = f[:int(len(magnitude)/2)]

    plt.figure(figsize=fig_size)
    plt.plot(left_f, left_spectrum)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")
    plt.show()

# STFT -> spectrogram
def STFT(sig):

    hop_length = 512  # 전체 frame 수
    n_fft = 2048  # frame 하나당 sample 수

# calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sr
    n_fft_duration = float(n_fft)/sr

# STFT
    stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

# take absolute value
    magnitude = np.abs(stft)

# magnitude > Decibels
    log_spectrogram = librosa.amplitude_to_db(magnitude)

# display spectrogram
    plt.figure(figsize=fig_size)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")

#mfccs
def mfccs(sig,a):
# extract 13 MFCCs
    hop_length = 512  # entire number of frame
    n_fft = 2048  # the number of sample per frame
    MFCCs = librosa.feature.mfcc(sig, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=a)

# display MFCCs
    plt.figure(figsize=fig_size)
    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    plt.show()

Waveform(sig)
fft(sig)
mfccs(sig,32)