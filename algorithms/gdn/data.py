import random
from torch.utils.data import DataLoader, Subset, Dataset
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config=None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]

        data = x_data
        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']
                                   ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        for i in rang:
            ft = data[:, i - slide_win:i]
            tar = data[:, i-1]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index

def get_loaders(train_dataset, seed, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)


    train_dataloader = DataLoader(train_subset, batch_size=batch,
                            shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                            shuffle=False)

    return train_dataloader, val_dataloader


def data_cleaning(dataframe):
    """ spectral residual 를 활용한 데이터 정제 """
    mag = []
    cols = dataframe.columns
    idx = dataframe.index

    for i in cols:
        mag.append(spectral_residual(dataframe[i]))
    mag_df = pd.DataFrame(np.column_stack(mag), columns=cols, index=idx)

    outliers_cols = mag_df.columns[(mag_df > 0.1).any()]
    for i in outliers_cols:
        indices = mag_df[mag_df[i] > 0.1].index
        dataframe[i].loc[indices] = np.nan  # dataframe[i].median()
        dataframe[i] = dataframe[i].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)

    return dataframe


def series_filter(values, k=3):
    if k >= len(values):
        k = len(values)

    res = np.cumsum(values, dtype=float)
    res[k:] = res[k:] - res[:-k]
    res[k:] = res[k:] / k

    for i in range(1, k):
        res[i] /= (i + 1)

    return res


def silency_map(values):
    eps = 1e-8
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

    eps_index = np.where(mag <= eps)[0]
    mag[eps_index] = eps
    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    # In our experiments, we set k = 7 for minutely time-series, k = 3 for hourly time-series and
    # k = 1 for daily time-series following the requirement of real application.

    spectral = np.exp(mag_log - series_filter(mag_log, k=60))
    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    sliency_map = np.fft.ifft(trans)
    return sliency_map


def spectral_residual(values):
    """
        values : 하나의 시계열 매트릭의 values (list, numpy array)
    """
    sliency_map = silency_map(values)
    spectral_residual = np.sqrt(sliency_map.real ** 2 + sliency_map.imag ** 2)

    return spectral_residual


def remove_outlier_func(df):
    dataframe = []
    cols = df.columns
    idx = df.index

    removed_negative_df = df.mask(df < 0, np.nan)

    for feat in removed_negative_df.columns:
        removed_negative_df[feat] = removed_negative_df[feat].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)
        feat_data = remove_global_outliers(removed_negative_df[feat].dropna().values)
        dataframe.append(feat_data)

    return pd.DataFrame(np.column_stack(dataframe), columns=cols)


def remove_global_outliers(feat_data):
    feat_data = np.log1p(feat_data.astype(float))
    mean_, std_ = feat_data.mean(), feat_data.std()

    outlier_idx = np.where((feat_data < (mean_ - 3 * std_)) | (feat_data > (mean_ + 3 * std_)))[0]
    feat_data[outlier_idx] = np.nan
    feat_series = pd.Series(feat_data)

    feat_series.interpolate(limit_area='inside', inplace=True)
    feat_series.interpolate(limit_direction='both', inplace=True)

    feat_data = np.expm1(feat_series.values)

    return feat_data
