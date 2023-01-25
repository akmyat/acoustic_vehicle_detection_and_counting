import numpy as np
import pandas as pd
import pickle

import librosa
from scipy import signal
import sounddevice as sd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy

# Audio sample rate
samplerate = 48000

# Down sample
downsample = 2

# Sample rate after downsampling
dsr = int(samplerate / downsample)

# Gain to apply to audio data in pre-processing
# 12dB for usual environment
# 0dB for very loud environment
# 24db for very quiet environment
input_gain_db = 12

# Input device
device = 'snd_rpi_i2s_card'

# Duration of each recorded sample in seconds
sample_duration = 20

# Time the microphone should record in advance before recording first sample
init_time = 5

# Time need BETWEEN DIFFERENT SAMPLES while recording
gap_time = 2

def butter_highpass(cutoff, fs, order=5):
    """
    Helper function for the high pass filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    HIgh-pass filter for digital audio data.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def set_gain_db(audiodata, gain_db):
    """
    This function allows to set the audio gain in decibel(dB).
    Values above 1 or below -1 are set to the max/min values.
    """
    audiodata *= np.power(10, gain_db / 10)
    return np.array([1 if s > 1 else -1 if s < -1 else s for s in audiodata], dtype=np.float32)

def process_audio_data(audiodata):
    """
    Remove the DC offset by applying a high-pass filter and
    increase the amplitude by setting a positive gain.
    """
    
    # Extract mono channels from input data.
    ch1 = np.array(audiodata[::downsample, 0], dtype=np.float32)
    ch2 = np.array(audiodata[::downsample, 1], dtype=np.float32)

    # High-pass filter the ata at a cutoff frequency of 10Hz
    ch1 = butter_highpass_filter(ch1, 10, dsr)
    ch2 = butter_highpass_filter(ch2, 10, dsr)

    # Amplify audio data.
    ch1 = set_gain_db(ch1, input_gain_db)
    ch2 = set_gain_db(ch2, input_gain_db)

    # Output the data in the same format as it came in.
    return np.array([[ch1[i], ch2[i]] for i in range(len(ch1))], dtype=np.float32)

def record_samples():
    global sample_duration, init_time
    rec_duration = init_time + sample_duration

    # Start the stereo recording.
    rec = sd.rec(int(rec_duration * samplerate), samplerate=samplerate, channels=2)
    return rec
    
if __name__ == "__main__":
    
    count = 0
    while True:
        rec = record_samples()
        processed = process_audio_data(rec)
        print(processed.shape)
        count += 1
        if count == 3:
            break