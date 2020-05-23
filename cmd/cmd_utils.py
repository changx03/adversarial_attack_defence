import logging
import os

from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, get_time_str


def set_logging(headername, dname, verbose, save_log):
    """Setting the logging configuration"""
    log_lvl = logging.DEBUG if verbose else logging.INFO
    time_str = get_time_str()
    if save_log:
        log_filename = f'{headername}_{dname}_{time_str}.log'
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
    elif dname == 'Synthetic':
        # No function call for synthetic
        return dc
    else:
        raise AttributeError('Received unknown dataset "{}"'.format(dname))
    return dc


def parse_model_filename(filename):
    """
    Parses the filename of a trained model. The filename should in
    "<model>_<dataset>_e<max epochs>[_<date>].pt" format'.
    """
    dirname = os.path.split(filename)
    arr = dirname[-1].split('_')
    model_name = arr[0]
    dataset_name = arr[1]
    return model_name, dataset_name
