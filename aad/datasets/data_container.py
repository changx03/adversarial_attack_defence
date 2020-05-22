
"""
This module implements the base class for DataContainer.
"""
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from scipy.io import arff
from torch.utils.data import DataLoader

from ..utils import (get_range, scale_normalize, shuffle_data,
                     swap_image_channel)
from .dataset_list import get_sample_mean, get_sample_std
from .generic_dataset import GenericDataset

logger = logging.getLogger(__name__)


class DataContainer:
    def __init__(self, dataset_dict, path=None):
        self.name = dataset_dict['name']
        self._data_type = dataset_dict['type']
        self._num_classes = int(dataset_dict['num_classes'])
        self._dim_data = dataset_dict['dim_data']
        self._path = path
        self._train_mean = None
        self._train_std = None
        self._dataframe = None
        self._data_range = None
        self._x_train_np = None
        self._y_train_np = None
        self._x_test_np = None
        self._y_test_np = None

        assert self._data_type in ('image', 'numeric')
        if path is not None:
            assert os.path.exists(path), f'{path} does NOT exist!'

    @property
    def data_type(self):
        return self._data_type

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def dim_data(self):
        return self._dim_data

    @property
    def path(self):
        return self._path

    @property
    def train_mean(self):
        return self._train_mean

    @property
    def train_std(self):
        return self._train_std

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def data_range(self):
        return self._data_range

    @property
    def x_train(self):
        return self._x_train_np

    @x_train.setter
    def x_train(self, x):
        self._x_train_np = x
        is_image = self._data_type == 'image'
        self._data_range = get_range(self._x_train_np, is_image=is_image)

        # update mean and mu
        self._train_mean = self._x_train_np.mean(axis=0)
        self._train_std = self._x_train_np.std(axis=0)

    @property
    def y_train(self):
        return self._y_train_np

    @y_train.setter
    def y_train(self, y):
        self._y_train_np = y

    @property
    def x_test(self):
        return self._x_test_np

    @x_test.setter
    def x_test(self, x):
        self._x_test_np = x

    @property
    def y_test(self):
        return self._y_test_np

    @y_test.setter
    def y_test(self, y):
        self._y_test_np = y

    @property
    def x_all(self):
        return np.vstack((self._x_train_np, self._x_test_np))

    @property
    def y_all(self):
        # for index encoding
        if len(self._y_train_np.shape) == 1:
            return np.append(self._y_train_np, self._y_test_np)
        # for one-hot encoding
        return np.vstack((self._y_train_np, self._y_test_np))

    def __len__(self):
        # total length = train + test
        return len(self._x_train_np) + len(self._x_test_np)

    def __call__(self, shuffle=True, normalize=False, size_train=0.8):
        """Load data and prepare for numpy arrays. `normalize` and `size_train`
        are not used in image datasets
        """
        since = time.time()
        if self._data_type == 'image':
            self._prepare_image_data(shuffle, num_workers=0)
            self._train_mean = get_sample_mean(self.name)
            self._train_std = get_sample_std(self.name)
        else:
            if self.name is not 'Synthetic':
                self._prepare_numeric_data(
                    shuffle, normalize, size_train)
                self._train_mean = self._x_train_np.mean(axis=0)
                self._train_std = self._x_train_np.std(axis=0)
            else:
                logger.warning('Load the synthetic data from external source before proceed.')

        time_elapsed = time.time() - since

        logger.info('Train size: %d - Test size: %d',
                    len(self._y_train_np), len(self._y_test_np))
        logger.info('Successfully load data. Time to complete: %dm %.3fs',
                    int(time_elapsed // 60), time_elapsed % 60)

    def get_dataloader(self,
                       batch_size=64,
                       is_train=True,
                       shuffle=True,
                       num_workers=0):
        """
        Returns a PyTorch DataLoader.
        """
        try:
            x_np = self._x_train_np if is_train else self._x_test_np
            y_np = self._y_train_np if is_train else self._y_test_np

            if self._data_type == 'image' and x_np.shape[1] not in (1, 3):
                x_np = swap_image_channel(x_np)

            dataset = GenericDataset(x_np, y_np)
            dataloader = DataLoader(
                dataset,
                batch_size,
                shuffle=shuffle,
                num_workers=num_workers)
            return dataloader
        except AttributeError:
            raise Exception('Call class instance first!')

    def _prepare_image_data(self, shuffle, num_workers):
        # for images, we prepare dataloader first, and then convert it to numpy array.
        dataset_train = self._get_dataset(train=True)
        dataset_test = self._get_dataset(train=False)

        batch_size = 128  # this batch size is only used for loading.

        # client should not access these loader directly
        dataloader_train = DataLoader(
            dataset_train,
            batch_size,
            shuffle=shuffle)
        dataloader_test = DataLoader(
            dataset_test,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

        self._x_train_np, self._y_train_np = self._loader_to_np(
            dataloader_train)
        self._x_test_np, self._y_test_np = self._loader_to_np(
            dataloader_test)
        # pytorch uses (c, h, w). numpy uses (h, w, c)
        self._x_train_np = swap_image_channel(self._x_train_np)
        self._x_test_np = swap_image_channel(self._x_test_np)

        self._data_range = get_range(self._x_train_np, is_image=True)

    def _prepare_numeric_data(self, shuffle, normalize, size_train):
        # for numeric, starts with a Pandas dataframe, and then
        # populate numpy array and then pytorch DataLoader
        self._dataframe = self._get_dataframe()
        m = self._dataframe.shape[1] - 1  # the label is also in the frame

        self._data_range = get_range(self._dataframe.values[:, :m])

        if shuffle:
            self._dataframe = shuffle_data(self._dataframe)

        x_train, y_train, x_test, y_test = self._split_dataframe2np(size_train)

        if normalize:
            (xmin, xmax) = self._data_range
            # NOTE: Carlini attack expects the data in range [0, 1]
            # mean = self._train_mean
            x_train = scale_normalize(x_train, xmin, xmax, mean=None)
            x_test = scale_normalize(x_test, xmin, xmax, mean=None)

        # to numpy array
        # NOTE: Only handle numeral data
        self._x_train_np = x_train.astype(np.float32)
        self._y_train_np = y_train.astype(np.long)
        self._x_test_np = x_test.astype(np.float32)
        self._y_test_np = y_test.astype(np.long)

    def _get_dataset(self, train):
        transform = tv.transforms.Compose([tv.transforms.ToTensor()])

        if self.name == 'MNIST':
            return tv.datasets.MNIST(
                self._path,
                train=train,
                download=True,
                transform=transform)
        elif self.name == 'CIFAR10':
            return tv.datasets.CIFAR10(
                self._path,
                train=train,
                download=True,
                transform=transform)
        elif self.name == 'SVHN':
            return tv.datasets.SVHN(
                self._path,
                split='train' if train else 'test',
                download=True,
                transform=transform)
        else:
            raise Exception(f'Dataset {self.name} not found!')

    def _loader_to_np(self, loader):
        n = len(loader.dataset)
        data_shape = tuple([n]+list(self._dim_data))
        label_shape = (n,)

        data_np = np.zeros(data_shape, dtype=np.float32)
        label_np = -np.ones(label_shape, dtype=np.int64)  # assign -1

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
        n = len(self._dataframe.index)
        num_train = size_train
        if size_train < 1:
            num_train = int(np.round(n * size_train))
        num_test = n - num_train
        m = self._dim_data[0]

        # these are numpy arrays
        x_train = self._dataframe.iloc[:num_train, :m].values
        y_train = self._dataframe.iloc[:num_train, -1].values
        x_test = self._dataframe.iloc[-num_test:, :m].values
        y_test = self._dataframe.iloc[-num_test:, -1].values

        # checking shapes
        assert x_train.shape == (num_train, m)
        assert y_train.shape == (num_train,)
        assert x_test.shape == (num_test, m)
        assert y_test.shape == (num_test,)

        return x_train, y_train, x_test, y_test

    def _check_file(self, file):
        logger.debug('Reading file %s', file)
        assert os.path.exists(file), f'{file} does NOT exist!'

    def _get_dataframe(self):
        if self.name == 'BankNote':
            file_path = os.path.join(
                self._path, 'data_banknote_authentication.txt')
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
            file_path = os.path.join(self._path, 'BreastCancerWisconsin.csv')
            self._check_file(file_path)
            return self._handle_bc_dataframe(file_path)
        elif self.name == 'WheatSeed':
            file_path = os.path.join(self._path, 'seeds_dataset.txt')
            self._check_file(file_path)
            return self._handle_wheat_seed_dataframe(file_path)
        elif self.name == 'HTRU2':
            file_path = os.path.join(self._path, 'HTRU2', 'HTRU_2.arff')
            self._check_file(file_path)
            return self._handle_htru2_dataframe(file_path)
        elif self.name == 'Iris':
            file_path = os.path.join(self._path, 'iris.data')
            self._check_file(file_path)
            return self._handle_iris_dataframe(file_path)
        else:
            raise Exception(f'Dataset {self.name} not found!')

    def _handle_bc_dataframe(self, file_path):
        """ Preprocessing the Breast Cancer Wisconsin (Diagnostic) DataFrame
        """
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

    @staticmethod
    def _handle_wheat_seed_dataframe(file_path):
        """ Preprocessing the Seeds of Wheat DataFrame
        """
        col_names = ['area', 'perimeter', 'compactness', 'kernel length',
                     'kernel width', 'asymmetry coefficient', 'kernel groove length',
                     'class']
        df = pd.read_csv(file_path, header=None, names=col_names, sep=r'\s+')
        # convert categorical data to integer codes
        df['class'] = df['class'].astype('category')
        # map [1, 2, 3] to [0, 1, 2]
        df['class'] = df['class'].cat.codes.astype('long')
        return df

    @staticmethod
    def _handle_htru2_dataframe(file_path):
        """ Preprocessing the HTRU2 DataFrame
        """
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        # convert categorical data to integer codes
        df['class'] = df['class'].astype('category')
        df['class'] = df['class'].cat.codes.astype('long')
        return df

    @staticmethod
    def _handle_iris_dataframe(file_path):
        """ Preprocessing the Iris DataFrame
        """
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
