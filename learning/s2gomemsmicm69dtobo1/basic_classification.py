import numpy as np
import pandas as pd
import pickle

import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy import signal

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy

# Audio sample config
samplerate = 48000                                         # Sample rate
downsample = 2                                                # Down sample
dsr = int(samplerate / downsample)          # Sample rate after downsampling
sample_duration = 1                                       # Duration of each recorded sample in seconds   
init_time = 5                                                      # Time the microphone should record in advance before recording a sample
prepare_time = 8                                             # Time needed BETWEEN DIFFERENT CLASSES
gap_time = 2                                                    # Time needed BETWEEN DIFFERENT SAMPLES
input_gain_db = 12                                        # Input gain in dB. 12 dB for usual environment, 0 dB for very loud environment, 24 dB for very quiet environment
# Device
device = 'snd_rpi_simple_card'

# Classifier config
classes = ['clap', 'snap', 'other']                     # Classes
samples_per_class = 10                                  # Number of samples per class

def butter_highpass(cutoff, fs, order=5):
    """
    Helper function for the highpass filter.
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    High-pass filter for digital audio data.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def set_gain_db(audiodata, gain_db):
    """
    This function allows to set the audio gain in decibel(dB).
    Values above 1 or below -1 are set to the max/min values.
    """
    audiodata *= np.power(10, gain_db/10)
    return np.array([1 if s > 1 else -1 if s < -1 else s for s in audiodata], dtype=np.float32)

def process_audio_data(audiodata):
    """
    Some basic input processing of the recorded audio data.
    Remove the DC offset by applying a high-pass filter and
    increase the amplitude by setting a positive gain.
    """

    # Extract mono channels from input data.
    ch1 = np.array(audiodata[::downsample, 0], dtype=np.float32)
    ch2 = np.array(audiodata[::downsample, 1], dtype=np.float32)

    # High-pass filter the data at a cutoff frequency of 10 Hz.
    ch1 = butter_highpass_filter(ch1, 10, dsr)
    ch2 = butter_highpass_filter(ch2, 10, dsr)

    # Amplify the audio data
    ch1 = set_gain_db(ch1, input_gain_db)
    ch2 = set_gain_db(ch2, input_gain_db)

    # Output the data in same format as it came in.
    return np.array([[ch1[i], ch2[2]] for i in range(len(ch1))], dtype=np.float32)

def record_samples():
    global init_time, prepare_time, gap_time

    # Calculate recording duration
    rec_duration = init_time + ((sample_duration + gap_time) * samples_per_class + prepare_time)  * len(classes)

    # Start the stereo recording
    rec = sd.rec(int(rec_duration * samplerate), samplerate=samplerate, channels=2, device=device)
    
    print("Recording started, BUT WAIT - give the microphone a bit time to settle. . .")
    sd.sleep(int(init_time * 1000))

    # Record samples for each class
    for cls in classes:
        print("Get ready to record samples for class '" + str(cls) +"'. . .")
        sd.sleep(int(prepare_time * 1000))

        for sample in range(samples_per_class):
            print("- RECORDING " + str(sample+1) + "/" +str(samples_per_class) + " -")
            sd.sleep(int(sample_duration * 1000))
            print("- STOP -")
            sd.sleep(int(gap_time * 1000))
    print("- DONE -")
    print("-" * 30)

    # Wait until recording is done.
    sd.wait()

    # Process the recorded audio data
    processed = process_audio_data(rec)
    print("Done.")
    return processed

def save_samples(recording):
    global init_time, prepare_time, gap_time
    """
    This function separates the audio recording to samples and saves them as Pickle file.
    """

    # Cut the start
    start_offset = init_time * dsr

    # Get the recorded samples
    samples = {}
    for cls in classes:
        samples[cls] = []
        start_offset += int(prepare_time * dsr)
        for i in range(samples_per_class):
            sample = recording[start_offset:start_offset+int(sample_duration * dsr)]
            samples[cls].append(sample)
            start_offset += int((sample_duration + gap_time) * dsr)
    pickle.dump(samples, open('samples.p', 'wb'))
    return samples

def generate_features(samples):
    """
    Generate features (X-inputs of Machine Learning model) out of the audio samples.
    Mel Frequency Cepstral Coefficients (MFCC) are used as features.
    """
    features = {}
    for cls in classes:
        features[cls] = []
        for sample in samples[cls]:
            # Generate MFCCs for left channel.
            mfcc_l = librosa.feature.mfcc(sample[:, 0], sr=dsr)
            # Generate MFCCs for right channel.
            mfcc_r = librosa.feature.mfcc(sample[:, 1], sr=dsr)
            features[cls].append(mfcc_l[0] + mfcc_r[0])
    return features

def features_to_dataframe(features):
    """
    Save the features and label in Pandas DataFrame.
    """
    Xy = []
    for label in samples:
        for feature in features[label]:
            row = [classes.index(label)]
            column_names = ['label']
            for idx, line in enumerate(feature):
                column_names.append('mfcc_l-' + str(idx))
                row.append(line)
                column_names.append('mfcc_r-' + str(idx))
                row.append(line)
            Xy.append(row)
    return pd.DataFrame(Xy, columns=column_names)

def load_samples():
    """
    Try to load previously saved samples. Trigger re-recording when no saved samples are available.
    """
    global samples_available
    try:
        samples = pickle.load(open('samples.p', 'rb'))
        samples_available = True
    except:
        samples = None
        samples_available = False
    return samples


if __name__ == "__main__":

    # Load samples or record new ones.
    samples = load_samples()

    if not samples_available:
        print("No sample file found - recording new one . . .")
        rec = record_samples()
        samples = save_samples(rec)
    else:
        print("Samples are loaded from 'samples.p'. Delete this file to record new audio samples.")
    
    features = generate_features(samples)
    df = features_to_dataframe(features)
    print(df)

    # Define features X as all columns of DataFrame except the label column.
    X = df.drop(['label'], axis='columns')
    y = df.label

    # Normalize X
    X = preprocessing.normalize(X, norm='l2')

    # Split X and y into training and testing data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Prepare the classifier
    model = MLPClassifier(solver='lbfgs', max_iter=10000)

    # Save the model to file
    pickle.dump(model, open('model.sav', 'wb'))

    # Train the classifier
    print("Start training . . .")
    model.fit(X_train, y_train)

    # Predict values with the trained model.
    y_pred = model.predict(X_test)

    # Evaluate the prediction performance using accuracy metric and print the result.
    score = accuracy(y_test, y_pred)
    print("Accuracy: " + str(score))