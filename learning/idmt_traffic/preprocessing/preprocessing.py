import os
import io
import glob
import zipfile
import pandas as pd
import numpy as np

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

from IPython.display import Audio, display


class IDMTTrafficDataset(Dataset):
    def __init__(self, X, y, zip_path, n_mels=None, target_sample_rate=22050):
        self.X = X
        self.y = y
        self.zfname = os.path.basename(zip_path).replace(".zip", "") + "/audio/"
        self.archive = zipfile.ZipFile(zip_path)
        self.target_sample_rate = target_sample_rate

        self.classes = y.unique()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        if n_mels == 16:            
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=16,
            )
        elif n_mels == 32:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=32,
            )
        elif n_mels == 64:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=64,
            )
        else:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=128,
            )
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal, sample_rate = self.__load_signal__(idx)
        signal = self.__mix_down_if_necessary__(signal)
        signal = self.__resample_if_necessary__(signal, sample_rate)
        mel_spec = self.mel_spec(signal)

        label = self.__get_label__(idx)
        
        return mel_spec, label

    def __load_signal__(self, idx):
        audio_filename = self.zfname + self.X.iloc[idx]['file'] + ".wav"
        signal, sample_rate = torchaudio.load(io.BytesIO(self.archive.read(audio_filename)))
        return signal, sample_rate
    
    def __mix_down_if_necessary__(self, signal):
        if signal.shape[0] > 1:
            signal = signal.mean(axis=0)
        return signal

    def __resample_if_necessary__(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            signal = T.Resample(sample_rate, self.target_sample_rate)(signal)
        return signal

    def __get_label__(self, idx):
        label = self.class_to_idx[self.y.iloc[idx]] 
        return label


class Preprocess:
    def __init__(self, zip_path):
        self.archive = zipfile.ZipFile(zip_path)
        annotation_path = os.path.basename(zip_path).replace(".zip", "") + "/annotation/"

        df_list = []

        fn_txt_list = [
            "idmt_traffic_all.txt",             # complete IDMT-Traffic dataset
            "eusipco_2021_train.txt",     # training set of EUSIPCO 2021 paper
            "eusipco_2021_test.txt"        # test set of EUSIPCO 2021 paper
        ]

        # import metadata
        for fn_txt in fn_txt_list:
            print("Metadata for {}: ".format(fn_txt))
            df_list.append(self.import_idmt_traffic_dataset(annotation_path + fn_txt))

        # Train Data
        self.train_df = df_list[1]

        # Test Data
        self.test_df = df_list[2]

    def train_val_test_split(self, val_size=0.2):
        X = self.train_df.drop(columns=['date_time', 'location', 'sample_pos', 'microphone', 'channel', 'vehicle'])
        y = self.train_df['vehicle']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        y_val.reset_index(drop=True, inplace=True)
        print(f"X_trian: {X_train.shape}\ty_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}\t\ty_val: {y_val.shape}")

        X_test = self.test_df.drop(columns=['date_time', 'location', 'sample_pos', 'microphone', 'channel', 'vehicle'])
        y_test = self.test_df['vehicle']
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        print(f"X_test: {X_test.shape}\ty_test: {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def import_idmt_traffic_dataset(self, fn_txt: str = "idmt_traffic_all") -> pd.DataFrame:
        """ Import IDMT-Traffic dataset
        Args:
            fn_txt (str): Text file with all WAV files
        Returns:
            df_dataset (pd.Dataframe): File-wise metadata
                Columns:
                    'file': WAV filename,
                    'is_background': True if recording contains background noise (no vehicle), False else
                    'date_time': Recording time (YYYY-MM-DD-HH-mm)
                    'location': Recording location
                    'speed_kmh': Speed limit at recording site (km/h), UNK if unknown,
                    'sample_pos': Sample position (centered) within the original audio recording,
                    'daytime': M(orning) or (A)fternoon,
                    'weather': (D)ry or (W)et road condition,
                    'vehicle': (B)us, (C)ar, (M)otorcycle, or (T)ruck,
                    'source_direction': Source direction of passing vehicle: from (L)eft or from (R)ight,
                    'microphone': (SE)= (high-quality) sE8 microphones, (ME) = (low-quality) MEMS microphones (ICS-43434),
                    'channel': Original stereo pair channel (12) or (34)
        """
        # load file list
        df_files = pd.read_csv(io.BytesIO(self.archive.read(fn_txt)), names=('file',))
        fn_file_list = df_files['file'].to_list()

        # load metadata from file names
        df_dataset = []

        for f, fn in enumerate(fn_file_list):
            fn = fn.replace('.wav', '')
            parts = fn.split('_')

            # background noise files
            if '-BG' in fn:
                date_time, location, speed_kmh, sample_pos, mic, channel = parts
                vehicle, source_direction, weather, daytime = 'None', 'None', 'None', 'None'
                is_background = True

            # files with vehicle passings
            else:
                date_time, location, speed_kmh, sample_pos, daytime, weather, vehicle_direction, mic, channel = parts
                vehicle, source_direction = vehicle_direction
                is_background = False

            channel = channel.replace('-BG', '')
            speed_kmh = speed_kmh.replace('unknownKmh', 'UNK')
            speed_kmh = speed_kmh.replace('Kmh', '')

            df_dataset.append({'file': fn,
                            'is_background': is_background,
                            'date_time': date_time,
                            'location': location,
                            'speed_kmh': speed_kmh,
                            'sample_pos': sample_pos,
                            'daytime': daytime,
                            'weather': weather,
                            'vehicle': vehicle,
                            'source_direction': source_direction,
                            'microphone': mic,
                            'channel': channel})

        df_dataset = pd.DataFrame(df_dataset, columns=('file', 'is_background', 'date_time', 'location', 'speed_kmh', 'sample_pos', 'daytime', 'weather', 'vehicle',
                                                    'source_direction', 'microphone', 'channel'))

        return df_dataset