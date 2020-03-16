import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from scipy.io import arff
from torch.utils.data import DataLoader, TensorDataset

from ..utils import (get_range, scale_normalize, shuffle_data,
                     swap_image_channel)
from .dataset_list import DATASET_LIST, get_sample_mean, get_sample_std
from .NumeralDataset import NumeralDataset


class DataContainer:
    def __init__(self, dataset_dict, path):
        self.name = dataset_dict['name']
        self.type = dataset_dict['type']
        assert self.type in ('image', 'quantitative')
        self.num_classes = int(dataset_dict['num_classes'])
        self.dim_data = dataset_dict['dim_data']
        self.path = path
        assert os.path.exists(path), f'{path} does NOT exist!'

    def __len__(self):
        # total length = train + test
        return len(self.data_train_np) + len(self.data_test_np)

    def __call__(self, shuffle=True, normalize=False,
                 size_train=0.8, enable_cross_validation=False):
        '''Load data and prepare for numpy arrays. `normalize` and `size_train` 
        are not used in image datasets
        '''
        # TODO: implement cross_validation
        assert enable_cross_validation == False, \
            'cross validation is not supported'

        print('Loading data...')
        since = time.time()
        if self.type == 'image':
            self._prepare_image_data(shuffle, num_workers=0)
            self.train_mean = get_sample_mean(self.name)
            self.train_std = get_sample_std(self.name)
        else:
            self._prepare_quantitative_data(
                shuffle, normalize, size_train)
            self.train_mean = self.data_train_np.mean(axis=0)
            self.train_std = self.data_train_np.std(axis=0)

        time_elapsed = time.time() - since

        print('Successfully load data! Time taken: {:2.0f}m {:3.1f}s'.format(
            time_elapsed // 60,
            time_elapsed % 60))

    def get_dataloader(self, batch_size=64, is_train=True, shuffle=True, num_workers=0):
        data_train_np = self.data_train_np
        data_test_np = self.data_test_np
        # pytorch uses (c, h, w). numpy uses (h, w, c)
        if self.type == 'image':
            data_train_np = swap_image_channel(self.data_train_np)
            data_test_np = swap_image_channel(self.data_test_np)

        if is_train:
            dataset = NumeralDataset(
                torch.as_tensor(data_train_np),
                torch.as_tensor(self.label_train_np))
        else:
            dataset = NumeralDataset(
                torch.as_tensor(data_test_np),
                torch.as_tensor(self.label_test_np))

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        return dataloader

    def _prepare_image_data(self, shuffle, num_workers):
        # for images, we prepare dataloader first, and then convert it to numpy array.
        print('Preparing DataLoaders...')
        self._dataset_train = self._get_dataset(train=True)
        self._dataset_test = self._get_dataset(train=False)

        batch_size = 128  # this batch size is only used for loading.
        self.dataloader_train = DataLoader(
            self._dataset_train,
            batch_size,
            shuffle=shuffle)
        self.dataloader_test = DataLoader(
            self._dataset_test,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

        print('Preparing numpy arrays...')
        self.data_train_np, self.label_train_np = self._loader_to_np(
            self.dataloader_train, train=True)
        self.data_test_np, self.label_test_np = self._loader_to_np(
            self.dataloader_test, train=False)
        # pytorch uses (c, h, w). numpy uses (h, w, c)
        self.data_train_np = swap_image_channel(self.data_train_np)
        self.data_test_np = swap_image_channel(self.data_test_np)

        self.data_range = get_range(self.data_train_np, is_image=True)

    def _prepare_quantitative_data(self, shuffle, normalize, size_train):
        # for quantitative, starts with a Pandas dataframe, and then
        # populate numpy array and then pytorch DataLoader
        print('Preparing DataFrame...')
        self.dataframe = self._get_dataframe()
        m = self.dataframe.shape[1] - 1  # the label is also in the frame

        self.data_range = get_range(self.dataframe.values[:, :m])

        print('Spliting train/test sets into numpy arrays...')
        if shuffle:
            self.dataframe = shuffle_data(self.dataframe)

        x_train, y_train, x_test, y_test = self._split_dataframe2np(size_train)

        if normalize:
            (xmin, xmax) = self.data_range
            x_train = scale_normalize(x_train, xmin, xmax)
            x_test = scale_normalize(x_test, xmin, xmax)

        # to numpy array
        # NOTE: Only handle numeral data
        self.data_train_np = x_train.astype(np.float32)
        self.label_train_np = y_train.astype(np.long)
        self.data_test_np = x_test.astype(np.float32)
        self.label_test_np = y_test.astype(np.long)

    def _get_dataset(self, train):
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        if self.name == 'MNIST':
            return tv.datasets.MNIST(
                self.path,
                train=train,
                download=True,
                transform=transform)
        elif self.name == 'CIFAR10':
            return tv.datasets.CIFAR10(
                self.path,
                train=train,
                download=True,
                transform=transform)
        elif self.name == 'SVHN':
            return tv.datasets.SVHN(
                self.path,
                split='train' if train else 'test',
                download=True,
                transform=transform)
        else:
            raise Exception(f'Dataset {self.name} not found!')

    def _loader_to_np(self, loader, train=True):
        n = len(loader.dataset)
        data_shape = tuple([n]+list(self.dim_data))
        label_shape = (n, )

        data_np = np.zeros(data_shape, dtype=np.float32)
        label_np = -np.ones(label_shape, dtype=np.long)  # assign -1

        start = 0
        with torch.no_grad():
            for x, y in loader:
                batch_size = len(x)
                data_np[start: start + batch_size] = x.numpy()
                label_np[start: start + batch_size] = y.numpy()
                start = start+batch_size
        return data_np, label_np

    def _split_dataframe2np(self, size_train):
        assert isinstance(size_train, (int, float))

        # split train/test by ratio of fixed size of train data
        n = len(self.dataframe.index)
        num_train = size_train
        if size_train < 1:
            num_train = int(np.round(n * size_train))
        num_test = n - num_train
        m = self.dim_data[0]

        # these are numpy arrays
        x_train = self.dataframe.iloc[:num_train, :m].values
        y_train = self.dataframe.iloc[:num_train, -1].values
        x_test = self.dataframe.iloc[-num_test:, :m].values
        y_test = self.dataframe.iloc[-num_test:, -1].values

        # checking shapes
        assert x_train.shape == (num_train, m)
        assert y_train.shape == (num_train,)
        assert x_test.shape == (num_test, m)
        assert y_test.shape == (num_test,)

        return x_train, y_train, x_test, y_test

    def _check_file(self, file):
        print(f'Reading from {file}')
        assert os.path.exists(file), f'{file} does NOT exist!'

    def _get_dataframe(self):
        if self.name == 'BankNote':
            file_path = os.path.join(
                self.path, 'data_banknote_authentication.txt')
            self._check_file(file_path)
            col_names = ['variance', 'skewness',
                         'curtosis', 'entropy', 'class']
            df = pd.read_csv(
                file_path,
                header=None,
                names=col_names,
                dtype=np.float32)
            # Expecting y (label, output) has column name "class"
            df['class'] = df['class'].astype('long')
            return df

        elif self.name == 'BreastCancerWisconsin':
            file_path = os.path.join(self.path, 'BreastCancerWisconsin.csv')
            self._check_file(file_path)
            return self._handle_bc_dataframe(file_path)
        elif self.name == 'WheatSeed':
            file_path = os.path.join(self.path, 'seeds_dataset.txt')
            self._check_file(file_path)
            return self._handle_wheat_seed_dataframe(file_path)
        elif self.name == 'HTRU2':
            file_path = os.path.join(self.path, 'HTRU2', 'HTRU_2.arff')
            self._check_file(file_path)
            return self._handle_htru2_dataframe(file_path)
        elif self.name == 'Iris':
            file_path = os.path.join(self.path, 'iris.data')
            self._check_file(file_path)
            return self._handle_iris_dataframe(file_path)
        else:
            raise Exception(f'Dataset {self.name} not found!')

    def _handle_bc_dataframe(self, file_path):
        ''' Preprocessing the Breast Cancer Wisconsin (Diagnostic) DataFrame
        '''
        df = pd.read_csv(file_path, index_col=0)

        # remove empty column
        df = df.drop(
            df.columns[df.columns.str.contains('^Unnamed')],
            axis=1)
        # rename column 'diagnosis' to 'class'
        df.rename({'diagnosis': 'class'}, axis='columns', inplace=True)
        # map categorical outputs to integer codes
        df['class'] = df['class'].astype('category')
        df['class'] = df['class'].cat.codes.astype('long')
        # move output column to the end of table
        col_names = df.columns
        col_names = [c for c in col_names if c != 'class'] + ['class']
        df = df[col_names]
        return df

    def _handle_wheat_seed_dataframe(self, file_path):
        ''' Preprocessing the Seeds of Wheat DataFrame
        '''
        col_names = ['area', 'perimeter', 'compactness', 'kernel length',
                     'kernel width', 'asymmetry coefficient', 'kernel groove length',
                     'class']
        df = pd.read_csv(file_path, header=None, names=col_names, sep='\s+')
        # convert categorical data to integer codes
        df['class'] = df['class'].astype('category')
        # map [1, 2, 3] to [0, 1, 2]
        df['class'] = df['class'].cat.codes.astype('long')
        return df

    def _handle_htru2_dataframe(self, file_path):
        ''' Preprocessing the HTRU2 DataFrame
        '''
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        # convert categorical data to integer codes
        df['class'] = df['class'].astype('category')
        df['class'] = df['class'].cat.codes.astype('long')
        return df

    def _handle_iris_dataframe(self, file_path):
        ''' Preprocessing the Iris DataFrame
        '''
        df = pd.read_csv(
            file_path,
            header=None,
            names=['SepalLength', 'SepalWidth',
                   'PetalLength', 'PetalWidth', 'class'],
            dtype={'SepalLength': np.float32,
                   'SepalWidth': np.float32,
                   'PetalLength': np.float32,
                   'PetalWidth': np.float32,
                   'class': np.str})
        df['class'] = df['class'].astype('category')
        df['class'] = df['class'].cat.codes.astype('long')
        return df