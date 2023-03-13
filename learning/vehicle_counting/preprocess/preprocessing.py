import os
import io
import glob
import zipfile
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torchvision.transforms import ToPILImage, ToTensor, Normalize

from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

from IPython.display import Audio, display

class VcdDataset(Dataset):
    def __init__(self, X, y, zip_path, n_mels=None, sample_length=20, target_sample_rate=22050, classes=15):
        self.X = X
        self.y = y
        self.sample_length = sample_length
        self.target_sample_rate = target_sample_rate
        self.num_samples = int(self.sample_length * self.target_sample_rate)
        
        self.zfname = os.path.basename(zip_path).replace(".zip", "") + "/"
        self.archive = zipfile.ZipFile(zip_path)

        self.classes = [i for i in range(classes)]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        if n_mels == 16:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=n_mels,
            )
        elif n_mels == 32:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=n_mels,
            )
        elif n_mels == 64:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_mels=n_mels,
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
        return signal, sample_rate        

    def __load_signal__(self, idx):
        audio_filename = self.zfname + self.X.iloc[idx]['file']
        signal, sample_rate = torchaudio.load(io.BytesIO(self.archive.read(audio_filename)))
        signal = self.__mix_down_if_necessary__(signal)
        signal = self.__resample_if_necessary__(signal, sample_rate)
        signal = self.__cut_if_necessary__(signal)
        signal = self.__right_pad_if_necessary__(signal)
        mel_spec = self.mel_spec(signal)
        log_mel_spec = self.__convert_to_db_scale__(mel_spec)
        img = self.__convert_to_image__(log_mel_spec)

        label = self.__get_label__(idx)
        return img, label

    def __mix_down_if_necessary__(self, signal):
        if signal.shape[0] > 1:
            signal = signal.mean(axis=0)
        return signal

    def __resample_if_necessary__(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            signal = T.Resample(sample_rate, self.target_sample_rate)(signal)
        return signal

    def __cut_if_necessary__(self, signal):
        if signal.size(0) > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def __right_pad_if_necessary__(self, signal):
        if signal.size(0) < self.num_samples:
            num_missing_samples = self.num_samples - signal.size(0)
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def __convert_to_db_scale__(self, mel_spec):
        log_mel_spec = F.amplitude_to_DB(mel_spec, 10, 1e-10, np.log10(max(mel_spec.max(), 1e-10)))
        return log_mel_spec
    
    def __convert_to_image__(self, log_mel_spec):
        eps  = 1e-6
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec_norm = (log_mel_spec - mean) / (std + eps)
        spec_min, spec_max = log_mel_spec_norm.min(), log_mel_spec_norm.max()
        img = 255 * (log_mel_spec_norm - spec_min) / (spec_max - spec_min)
        img = ToPILImage()(img).convert('RGB')
        img_tensor = ToTensor()(img)
        normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        return normalize_img

    def __get_label__(self, idx):
        label = self.class_to_idx[self.y.iloc[idx]] 
        return label

class Preprocess:
    def __init__(self, train_zip_path, test_zip_path):
        self. X_train, self.y_train = self.load_data(train_zip_path)
        self.X_test, self.y_test = self.load_data(test_zip_path)
        
        self.train_df = pd.DataFrame(self.X_train, columns=['file'])
        self.train_df['vehicle_count'] = self.y_train

        self.test_df = pd.DataFrame(self.X_test, columns=['file'])
        self.test_df['vehicle_count'] = self.y_test

    def load_data(self, zip_path):
        zfname = os.path.basename(zip_path).replace('.zip', '') + "/"
        archive = zipfile.ZipFile(zip_path, 'r')
        X = [os.path.basename(file) for file in archive.namelist() if '.wav' in file]

        y = []
        vc_files = [os.path.basename(file) for file in archive.namelist() if '.txt' in file]
        for file in vc_files:
            vehicle_count = self.get_vehicle_count(archive, zfname + file)
            y.append(vehicle_count)
        return X, y
    
    def get_vehicle_count(self, zipfile, filename):
        with zipfile.open(filename) as f:
            minima = f.readlines()
        minima_positions = np.array([float(x.strip()) for x in minima])
        vehicle_count = 0
        if minima_positions[0] >= 0:
            vehicle_count = minima_positions.size
        return vehicle_count

    def train_val_test_split(self, val_size=0.2):
        X = self.train_df.drop(columns=['vehicle_count'])
        y = self.train_df['vehicle_count']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        y_val.reset_index(drop=True, inplace=True)
        print(f"X_trian: {X_train.shape}\ty_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}\t\ty_val: {y_val.shape}")

        X_test = self.test_df.drop(columns=['vehicle_count'])
        y_test = self.test_df['vehicle_count']
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        print(f"X_test: {X_test.shape}\ty_test: {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test        

    def train_test_split(self):
        X_train = self.train_df.drop(columns=['vehicle_count'])
        y_train = self.train_df['vehicle_count']
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        print(f"X_trian: {X_train.shape}\ty_train: {y_train.shape}")

        X_test = self.test_df.drop(columns=['vehicle_count'])
        y_test = self.test_df['vehicle_count']
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        print(f"X_test: {X_test.shape}\ty_test: {y_test.shape}")

        return X_train, X_test, y_train, y_test