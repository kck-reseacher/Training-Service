import os
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1, predict=False):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.predict = predict

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        if not self.predict:
            return (len(self.data) - self.window - self.horizon + 1)//2  # 한 epoch 당 학습 데이터 반만 사용
        else:
            return len(self.data) - self.window - self.horizon + 1

def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


class Aug:
    def augment_anomaly_data(X, y, augmentation_factor=40):
        X_augmented, y_augmented = [], []

        for i in range(len(y)):
            if torch.any(y[i]):
                for _ in range(augmentation_factor):
                    augmented_data = Aug.apply_noise_augmentation(X[i])
                    X_augmented.append(augmented_data)
                    y_augmented.append(y[i])

        if len(X_augmented) == 0:
            return X, y
        else:
            X_augmented = torch.cat([X, torch.stack(X_augmented)], dim=0)
            y_augmented = torch.cat([y, torch.stack(y_augmented)], dim=0)
            return X_augmented, y_augmented

    def apply_noise_augmentation(data, noise_factor=0.002):
        noise = torch.randn_like(data) * noise_factor
        augmented_data = data + data * noise

        return augmented_data


def split_series(data, n_past, n_future):
    X, y = [], []
    for window_start in range(len(data)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(data):
            break
        past, future = np.array(data[window_start:past_end, :]), np.array(data[past_end:future_end, :])
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


def filter_window(X, y, window_size=60, threshold=0.06):
    X_filtered, y_filtered = [], []

    if len(y.shape) == 2:
        is_one = np.any(y, axis=1)
    else:
        is_one = y

    for i in range(X.shape[0] - window_size - 1):
        window = X[i]
        anomalies = np.sum(np.array(is_one[i - window_size:i]))
        ratio = anomalies / window_size
        if ratio < threshold:
            X_filtered.append(window)
            y_filtered.append(y[i])

    return np.array(X_filtered), np.array(y_filtered)
