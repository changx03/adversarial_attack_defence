import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from scipy.io import arff
from torch.utils.data import DataLoader, TensorDataset

from .dataset_list import DATASET_LIST, get_sample_mean, get_sample_std
from .NumeralDataset import NumeralDataset
from .utils import get_range, scale_normalize, shuffle_data


class DataContainer:
    def __init__(self, dataset_dict, path):
        self.name = dataset_dict['name']
        assert self.name in DATASET_LIST['image'].keys() \
            or self.name in DATASET_LIST['quantitative'].keys()
        self.type = dataset_dict['type']
        assert self.type in ('image', 'quantitative')
        self.size = int(dataset_dict['size'])
        self.num_classes = int(dataset_dict['num_classes'])
        self.dim_data = dataset_dict['dim_data']
        self.path = path
        assert os.path.exists(path), f'{path} does NOT exist!'

    def __len__(self):
        return self.size

    def __call__(self, batch_size=16, transform=None, shuffle=True,
                 normalize=False, num_workers=0, size_train=0.8,
                 require_np_array=True, enable_cross_validation=False):
        # TODO: implement cross_validation
        assert enable_cross_validation == False, \
            'cross validation is not supported'

        self.batch_size = batch_size
        self.transform = transform if transform is not None \
            else tv.transforms.Compose([tv.transforms.ToTensor()])

        print('Loading data...')
        since = time.time()
        self._prepare_data(
            shuffle, normalize, num_workers, size_train, require_np_array)

        if self.type == 'image':
            self.train_mean = get_sample_mean(self.name)
            self.train_std = get_sample_std(self.name)
        else:
            self.train_mean = self.data_train_np.mean(axis=0)
            self.train_std = self.data_train_np.std(axis=0)

        time_elapsed = time.time() - since

        assert self.num_train != 0 and self.num_test != 0, \
            'WARNING: empty dataset!'
        assert self.num_train + self.num_test == self.size, \
            'WARNING: train+test is NOT equal to data size'

        print('Successfully load data! Time taken: {:2.0f}m {:3.1f}s'.format(
            time_elapsed // 60,
            time_elapsed % 60))

    def _prepare_image_data(self, shuffle, num_workers, require_np_array):
        # for images, we prepare dataloader first, and then convert it to numpy array.
        print('Preparing DataLoaders...')
        self._dataset_train = self._get_dataset(True)
        self._dataset_test = self._get_dataset(False)

        self.dataloader_train = DataLoader(
            self._dataset_train,
            self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        self.dataloader_test = DataLoader(
            self._dataset_test,
            self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

        # get dimensions
        self.num_train = len(self._dataset_train)
        self.num_test = len(self._dataset_test)
        self.dim_train = tuple([self.num_train] + list(self.dim_data))
        self.dim_test = tuple([self.num_test] + list(self.dim_data))

        # to numpy array
        if require_np_array:
            print('Preparing numpy arrays...')
            self.data_train_np, self.label_train_np = self._loader_to_np(
                self.dataloader_train, train=True)
            self.data_test_np, self.label_test_np = self._loader_to_np(
                self.dataloader_test, train=False)

    def _prepare_quantitative_data(self, shuffle, normalize,
                                   num_workers, size_train, require_np_array):
        # for quantitative, starts with a Pandas dataframe, and then
        # populate numpy array and then pytorch DataLoader
        print('Preparing DataFrame...')
        self.dataframe = self._get_dataframe()
        n = len(self.dataframe.index)
        m = self.dataframe.shape[1] - 1  # the label is also in the frame

        assert self.size == n, f'Expecting size {self.size}, got {n}'
        assert self.dim_data[0] == m, \
            f'Expecting {self.dim_data[0]} attributes, got {m}'

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
        self.data_train_np = x_train
        self.label_train_np = y_train
        self.data_test_np = x_test
        self.label_test_np = y_test

        # record dimensions
        self.num_train = len(x_train)
        self.num_test = len(x_test)
        self.dim_train = x_train.shape
        self.dim_test = x_test.shape

        # to pytorch DataLoader
        print('Preparing DataLoaders...')
        self._dataset_train = NumeralDataset(
            torch.Tensor(x_train),
            torch.Tensor(y_train))
        self._dataset_test = NumeralDataset(
            torch.Tensor(x_test),
            torch.Tensor(y_test))

        self.dataloader_train = DataLoader(
            self._dataset_train,
            self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        self.dataloader_test = DataLoader(
            self._dataset_test,
            self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

    def _prepare_data(self, shuffle, normalize,
                      num_workers, size_train, require_np_array):
        if self.type == 'image':
            self._prepare_image_data(shuffle, num_workers, require_np_array)
        elif self.type == 'quantitative':
            self._prepare_quantitative_data(
                shuffle, normalize, num_workers, size_train, require_np_array)
        else:
            # TODO: expecting mixed data in future
            raise Exception('Not implemented!')

    def _get_dataset(self, train):
        if self.name == 'MNIST':
            return tv.datasets.MNIST(
                self.path,
                train=train,
                download=True,
                transform=self.transform)
        elif self.name == 'CIFAR10':
            return tv.datasets.CIFAR10(
                self.path,
                train=train,
                download=True,
                transform=self.transform)
        elif self.name == 'SVHN':
            return tv.datasets.SVHN(
                self.path,
                split='train' if train else 'test',
                download=True,
                transform=self.transform)
        else:
            raise Exception(f'Dataset {self.name} not found!')

    def _loader_to_np(self, loader, train):
        data_shape = self.dim_train if train else self.dim_test
        label_shape = (self.num_train if train else self.num_test,)

        data_np = np.zeros(data_shape, dtype=np.float32)
        # assign -1 for all labels
        label_np = -np.ones(label_shape, dtype=np.int64)

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
        n = self.size
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
            df['class'] = df['class'].astype('int64')
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
        df['class'] = df['class'].cat.codes.astype('int64')
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
        df['class'] = df['class'].cat.codes.astype('int64')
        return df

    def _handle_htru2_dataframe(self, file_path):
        ''' Preprocessing the HTRU2 DataFrame
        '''
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        # convert categorical data to integer codes
        df['class'] = df['class'].astype('category')
        df['class'] = df['class'].cat.codes.astype('int64')
        return df

    def _check_file(self, file):
        print(f'Reading from {file}')
        assert os.path.exists(file), f'{file} does NOT exist!'
