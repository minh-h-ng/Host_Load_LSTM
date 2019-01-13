import numpy as np
import pandas as pd
import torch.utils.data as data
import logging
import math

from sklearn import preprocessing

# Initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class HostDataset(data.Dataset):
    # dataset_type = 'train' / 'validation' / 'test'
    def __init__(self, filename, window_size, dataset_type, n_test_days, output_length, hosts_per_group, scaler=None):
        self.window_size = window_size
        self.output_length = output_length
        self.hosts_per_group = hosts_per_group

        # load dataset as float64 (required by scaler)
        #self.dataset = pd.read_csv(filename, header=0, usecols=[0], names=['CPU_Usage'], dtype=np.float64).values

        self.dataset = pd.read_csv(filename, dtype=np.float64).values
        #self.dataset = self.dataset.values.reshape(-1).reshape(-1,1)

        self.length_per_host = self.dataset.shape[0]

        n_train_days = 29 - n_test_days

        if dataset_type == 'train':
            self.split_length = int(n_train_days * 288 / window_size)
            self.total_length = self.split_length * self.hosts_per_group
            self.startPoint = window_size
        elif dataset_type == 'test':
            self.split_length = int((n_test_days-1) * 288 / window_size + (288 - output_length - window_size) / window_size + 1)
            self.total_length = self.split_length * self.hosts_per_group
            self.startPoint = n_train_days * 288 + window_size
        else:
            assert False

        # normalize
        self.scaler = scaler
        to_fit = self.dataset[: 288 * n_train_days][:]
        self.scaler.fit_transform(to_fit.T.reshape(-1).reshape(-1,1))
        #self.scaler.fit_transform(self.dataset[: 288 * n_train_days][:])

        self.dataset = self.dataset.T.reshape(-1).reshape(-1, 1)
        self.dataset = self.scaler.transform(self.dataset).T.reshape(-1)

        # cast to float32 (required by pytorch)
        self.dataset = self.dataset.astype(np.float32)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):

        host_number = int(index / self.split_length)
        index_in_host = index % self.split_length

        x = self.dataset[host_number * self.length_per_host + self.startPoint + index_in_host * self.window_size - self.window_size:
                         host_number * self.length_per_host + self.startPoint + index_in_host * self.window_size]
        y = self.dataset[host_number * self.length_per_host + self.startPoint + index_in_host * self.window_size:
                         host_number * self.length_per_host + self.startPoint + index_in_host * self.window_size + self.output_length]

        return x, y

    def get_scaler(self):
        return self.scaler