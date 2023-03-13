
# Python code to get mel spectrogram with librosa

```python
mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=window_len, hop_length=hop_len, win_length=window_len, window=hann, center=True, n_mels=mel_bands_num, fmin=f_min, fmax=f_max, norm='slaney')
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
log_mel_spectrogram.shape
```

# Python code to preprocess audio with Tensorflow

```python
def _load_signal(audio_filename):
    audio = tfio.audio.AudioIOTensor(audio_filename)
    signal, sample_rate = audio.to_tensor(), int(audio.rate)
    return signal, sample_rate

def _mix_if_necessary(signal):
    if len(signal.shape) == 2:
        signal = tf.math.reduce_mean(signal, axis=1)
    return signal

def _cut_if_necessary(signal):
    if len(signal) / sample_rate > signal_len:
        signal = signal[:int(sample_rate * signal_len)]
    return signal

def _pad_if_necessary(signal):
    if len(signal) / sample_rate < signal_len:
        paddings = int(signal_len * sample_rate - signal.size)
        signal = tf.pad(signal, [[paddings, paddings]], mode='constant')
    return signal

def _resample_if_necessary(signal, sample_rate):
    if sample_rate != 44100:
        signal = tfio.audio.resample(signal, sample_rate, 44100)
    return signal

def _get_num_of_frames(signal, frame_len, hop_len):
    return int((len(signal) - frame_len) / hop_len) + 1

def _get_mel_transform_mtx(sample_rate, n_fft, n_mels, fmin, fmax):
    num_spectrogram_bins = n_fft // 2 +1
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels, 
        num_spectrogram_bins=num_spectrogram_bins, 
        sample_rate=sample_rate, 
        lower_edge_hertz=fmin, 
        upper_edge_hertz=fmax)

item = 0
audio_filename = audio_files[item]

window = tf.signal.hamming_window(window_len)

signal, sample_rate = _load_signal(audio_filename)
signal = _mix_if_necessary(signal)
signal = _cut_if_necessary(signal)
signal = _pad_if_necessary(signal)
signal = _resample_if_necessary(signal, sample_rate)

paddings = int(window_len / 2)
signal = tf.pad(signal, [[paddings, paddings]], mode='constant')
nb_frames = _get_num_of_frames(signal, window_len, hop_len)

mel_transform_mtx = _get_mel_transform_mtx(sample_rate, window_len, mel_bands_num, f_min, f_max)
```