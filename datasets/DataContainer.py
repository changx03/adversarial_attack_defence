import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset

DATASET_LIST = {
    'image': {
        'MNIST': {
            'name': 'MNIST',
            'type': 'image',
            'size': 7e4,
            'num_classes': 10,
            'sample_dim': (1, 28, 28),
        },
        'CIFAR10': {
            'name': 'CIFAR10',
            'type': 'image',
            'size': 6e4,
            'num_classes': 10,
            'dim': (3, 32, 32),
        },
        'SVHM': {
            'name': 'SVHM',
            'type': 'image',
            'size': 73257 + 26032,
            'num_classes': 10,
            'sample_dim': (3, 32, 32),
        },
    },
    'quantitative': {
        'BankNote': {
            'name': 'BankNote',
            'type': 'quantitative',
            'size': 1372,
            'num_classes': 2,
            'sample_dim': (4),
        },
        'BreastCancerWisconsin': {
            'name': 'BreastCancerWisconsin',
            'type': 'quantitative',
            'size': 569,
            'num_classes': 2,
            'sample_dim': (30),
        },
        'WheatSeed': {
            'name': 'WheatSeed',
            'type': 'quantitative',
            'size': 210,
            'num_classes': 3,
            'sample_dim': (7),
        },
        'HTRU2': {
            'name': 'HTRU2',
            'type': 'quantitative',
            'size': 17898,
            'num_classes': 2,
            'sample_dim': (8),
        },
    },
}


class DataContainer:
    def __init__(self, dataset_dict, path):
        self.name = dataset_dict['name']
        assert self.name in DATASET_LIST['image'].keys() \
            or self.name in DATASET_LIST['quantitative'].keys()
        self.type = dataset_dict['type']
        assert self.type in ('image', 'quantitative')
        self.size = int(dataset_dict['size'])
        self.num_classes = int(dataset_dict['num_classes'])
        self.sample_dim = dataset_dict['sample_dim']

        if self.type == 'image':
            self.c, self.h, self.w = self.sample_dim

        self.path = path
        assert isinstance(self.path, str)

    def __len__(self):
        return self.size

    def __call__(self, batch_size,
                 transform=None, shuffle=True,
                 num_workers=0, size_train=0.8,
                 require_np_array=True, enable_cross_validation=False):
        # TODO: implement cross_validation
        assert enable_cross_validation == False, \
            'cross validation is not supported'

        self.batch_size = batch_size
        self.transform = transform if transform is not None \
            else tv.transforms.Compose([tv.transforms.ToTensor()])

        # pytorch dataloaders
        print('Preparing dataloaders...')
        self._dataset_train = self._get_dataset(True)
        self._dataset_test = self._get_dataset(False)

        self.dataloader_train = DataLoader(
            self._dataset_train,
            self.batch_size,
            shuffle=True,
            num_workers=num_workers)
        self.dataloader_test = DataLoader(
            self._dataset_test,
            self.batch_size,
            shuffle=True,
            num_workers=num_workers)

        # make dimension variables
        self.num_train = len(self._dataset_train)
        self.num_test = len(self._dataset_test)
        self.train_dim = tuple([self.num_train] + list(self.sample_dim))
        self.test_dim = tuple([self.num_test] + list(self.sample_dim))

        # numpy arrays
        print('Preparing numpy arrays...')
        self.data_train_np, self.label_train_np = self._loader_to_np(
            self.dataloader_train, train=True)
        self.data_test_np, self.label_test_np = self._loader_to_np(
            self.dataloader_test, train=False)

        print('Successfully loaded data')

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
        data_dim = self.train_dim if train else self.test_dim
        label_dim = (self.num_train if train else self.num_test,)

        data_np = np.zeros(data_dim, dtype=np.float32)
        # assign -1 for all labels
        label_np = -np.ones(label_dim, dtype=np.int64)

        start = 0
        with torch.no_grad():
            for x, y in loader:
                batch_size = len(x)
                data_np[start: start + batch_size] = x.numpy()
                label_np[start: start + batch_size] = y.numpy()
                start = start+batch_size
        return data_np, label_np
