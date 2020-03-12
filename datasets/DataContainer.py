import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from .NumeralDataset import NumeralDataset
import os
import time

DATASET_LIST = {
    'image': {
        'MNIST': {
            'name': 'MNIST',
            'type': 'image',
            'size': 7e4,
            'num_classes': 10,
            'dim_data': (1, 28, 28),
        },
        'CIFAR10': {
            'name': 'CIFAR10',
            'type': 'image',
            'size': 6e4,
            'num_classes': 10,
            'dim_data': (3, 32, 32),
        },
        'SVHM': {
            'name': 'SVHM',
            'type': 'image',
            'size': 73257 + 26032,
            'num_classes': 10,
            'dim_data': (3, 32, 32),
        },
    },
    'quantitative': {
        'BankNote': {
            'name': 'BankNote',
            'type': 'quantitative',
            'size': 1372,
            'num_classes': 2,
            'dim_data': (4,),
        },
        'BreastCancerWisconsin': {
            'name': 'BreastCancerWisconsin',
            'type': 'quantitative',
            'size': 569,
            'num_classes': 2,
            'dim_data': (30,),
        },
        'WheatSeed': {
            'name': 'WheatSeed',
            'type': 'quantitative',
            'size': 210,
            'num_classes': 3,
            'dim_data': (7,),
        },
        'HTRU2': {
            'name': 'HTRU2',
            'type': 'quantitative',
            'size': 17898,
            'num_classes': 2,
            'dim_data': (8,),
        },
    },
}


def get_range(data):
    '''return (min, max) of a numpy array
    '''
    assert type(data) == np.ndarray

    x_max = np.max(data, axis=0)
    x_min = np.min(data, axis=0)
    return (x_min, x_max)


def scale_normalize(data, xmin, xmax):
    ''' scaling normalization puts data in range between 0 and 1
    '''
    assert (type(data) == np.ndarray and
            type(xmax) == np.ndarray and
            type(xmin) == np.ndarray)
    assert data.shape[1] == len(xmax) and data.shape[1] == len(xmin)

    return (data - xmin) / (xmax - xmin)


def shuffle_data(data):
    assert isinstance(data, (np.ndarray, pd.DataFrame))

    if isinstance(data, np.ndarray):
        n = len(data)
        shuffled_indices = np.random.permutation(n)
        return data[shuffled_indices]
    else:
        n = len(data.index)
        shuffled_indices = np.random.permutation(n)
        return data.iloc[shuffled_indices]


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

        if self.type == 'image':
            self.c, self.h, self.w = self.dim_data

        self.path = path
        assert isinstance(self.path, str)

    def __len__(self):
        return self.size

    def __call__(self, batch_size, transform=None, shuffle=True, normalize=False,
                 num_workers=0, size_train=0.8,
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
        time_elapsed = time.time() - since
        print('Successfully loaded data - {:2.0f}m {:3.1f}s'.format(
            time_elapsed // 60,
            time_elapsed % 60))

    def _prepare_image_data(self, shuffle, num_workers, require_np_array):
        # for images, we prepare dataloader first, and then convert it to numpy array.
        # pytorch dataloaders
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
        assert self.num_train != 0 and self.num_test != 0, 'WARNING: empty dataset!'
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
        m = self.dataframe.shape[1] - 1  # the label is also insided frame

        assert self.size == n, f'Expecting size {self.size}, got {n}'
        assert self.dim_data[0] == m, \
            f'Expecting {self.dim_data[0]} attributes, got {m}'

        # Expecting y/class/label has column name "class"
        self.dataframe['class'] = self.dataframe['class'].astype('int64')
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
            # TODO: implement other datasets
            raise Exception('Not implemented!')

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
            print(f'Reading from {file_path}')
            if not os.path.exists(file_path):
                raise FileExistsError(
                    'data_banknote_authentication.txt does NOT exist!')

            dataframe = pd.read_csv(
                file_path,
                header=None,
                names=['variance', 'skewness', 'curtosis', 'entropy', 'class'],
                dtype=np.float32)
            return dataframe
        else:
            # TODO: implement dataframe for other sets
            raise Exception('Not implemented!')
