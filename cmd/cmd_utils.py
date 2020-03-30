import logging
import os

from aad.utils import get_time_str, get_data_path
from aad.datasets import DATASET_LIST, DataContainer

def set_logging(dname, verbose, save_log):
    """Setting the logging configuration"""
    log_lvl = logging.DEBUG if verbose else logging.INFO
    time_str = get_time_str()
    if save_log:
        log_filename = f'train_{dname}_{time_str}.log'
        if not os.path.exists('log'):
            os.makedirs('log')
        logging.basicConfig(
            filename=os.path.join('log', log_filename),
            format='%(asctime)s:%(levelname)s:%(module)s:%(message)s',
            level=log_lvl)
    else:
        logging.basicConfig(level=log_lvl)

def get_data_container(dname, use_shuffle=True, use_normalize=True):
    """Returns a DataContainer based on given name"""
    dataset = DATASET_LIST[dname]
    dc = DataContainer(dataset, get_data_path())
    if dname in ('MNIST', 'CIFAR10', 'SVHN'):
        dc(shuffle=use_shuffle)
    elif dname == 'Iris':
        dc(shuffle=use_shuffle, normalize=use_normalize, size_train=0.6)
    elif dname in ('BankNote', 'BreastCancerWisconsin', 'HTRU2', 'WheatSeed'):
        dc(shuffle=use_shuffle, normalize=use_normalize)
    else:
        raise AttributeError('Received unknown dataset "{}"'.format(dname))
    return dc
