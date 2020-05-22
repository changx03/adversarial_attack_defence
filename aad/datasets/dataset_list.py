"""
This module contains the lookups for avaliable datasets
"""
import logging

logger = logging.getLogger(__name__)

DATASET_LIST = {
    'MNIST': {
        'name': 'MNIST',
        'type': 'image',
        'size': int(7e4),
        'num_classes': 10,
        'dim_data': (1, 28, 28),
    },
    'CIFAR10': {
        'name': 'CIFAR10',
        'type': 'image',
        'size': int(6e4),
        'num_classes': 10,
        'dim_data': (3, 32, 32),
    },
    'SVHN': {
        'name': 'SVHN',
        'type': 'image',
        'size': 73257 + 26032,
        'num_classes': 10,
        'dim_data': (3, 32, 32),
    },
    'BankNote': {
        'name': 'BankNote',
        'type': 'numeric',
        'size': 1372,
        'num_classes': 2,
        'dim_data': (4,),
    },
    'BreastCancerWisconsin': {
        'name': 'BreastCancerWisconsin',
        'type': 'numeric',
        'size': 569,
        'num_classes': 2,
        'dim_data': (30,),
    },
    'HTRU2': {
        'name': 'HTRU2',
        'type': 'numeric',
        'size': 17898,
        'num_classes': 2,
        'dim_data': (8,),
    },
    'Iris': {
        'name': 'Iris',
        'type': 'numeric',
        'size': 150,
        'num_classes': 3,
        'dim_data': (4,),
    },
    'WheatSeed': {
        'name': 'WheatSeed',
        'type': 'numeric',
        'size': 210,
        'num_classes': 3,
        'dim_data': (7,),
    }}


def get_dataset_list():
    return list(DATASET_LIST.keys())


# Only images are listed here. For numeric data, the mean and std will vary
# by the normalization option.
MEAN_LOOKUP = {
    'MNIST': [0.13066046],
    'CIFAR10': [0.49139947, 0.48215836, 0.44653094],
    'SVHN': [0.43768215, 0.4437698, 0.47280422],
}


STD_LOOKUP = {
    'MNIST': [0.30150425],
    'CIFAR10': [0.20230092, 0.1994128, 0.20096162],
    'SVHN': [0.12008651, 0.12313706, 0.10520393],
}


def get_sample_mean(dataset_name):
    if dataset_name in MEAN_LOOKUP.keys():
        return MEAN_LOOKUP[dataset_name]
    logger.warning('%s is not in the lookup', dataset_name)
    return [0.]


def get_sample_std(dataset_name):
    if dataset_name in MEAN_LOOKUP.keys():
        return STD_LOOKUP[dataset_name]
    logger.warning('%s is not in the lookup', dataset_name)
    return [1.]


def get_synthetic_dataset_dict(size, num_classes, num_attributes):
    """
    Get the dictionary of the synthetic dataset.
    """
    return {
        'name': 'Synthetic',
        'type': 'numeric',
        'size': size,
        'num_classes': num_classes,
        'dim_data': (num_attributes,),
    }
