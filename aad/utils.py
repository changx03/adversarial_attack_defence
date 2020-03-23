"""
This module contains all static utility functions
"""
import datetime
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def master_seed(seed):
    """
    Set the seed for all random number generators used in the library. This 
    ensures experiments reproducibility and stable testing.
    :param seed: The value to be seeded in the random number generators.
    :type seed: `int`
    """
    import numbers
    import random

    if not isinstance(seed, numbers.Integral):
        raise TypeError(
            'The seed for random number generators has to be an integer.')

    # Set Python seed
    random.seed(seed)

    # Set Numpy seed
    np.random.seed(seed)

    # Now try to set seed for all specific frameworks
    try:
        import torch

        logger.debug('Setting random seed for PyTorch.')
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.warning('Could not set random seed for PyTorch.')


def get_range(data, is_image=False):
    """return (min, max) of a numpy array
    """
    assert type(data) == np.ndarray

    axis = None if is_image else 0
    x_max = np.max(data, axis=axis)
    x_min = np.min(data, axis=axis)
    return (x_min, x_max)


def scale_normalize(data, xmin, xmax):
    """ scaling normalization puts data in range between 0 and 1
    """
    assert (type(data) == np.ndarray and
            type(xmax) == np.ndarray and
            type(xmin) == np.ndarray)
    assert data.shape[1] == len(xmax) and data.shape[1] == len(xmin)

    return (data - xmin) / (xmax - xmin)


def scale_unnormalize(data, xmin, xmax):
    """rescaling the normalized data back to raw
    """
    assert (type(data) == np.ndarray and
            type(xmax) == np.ndarray and
            type(xmin) == np.ndarray)
    assert data.shape[1] == len(xmax) and data.shape[1] == len(xmin)
    assert np.all(np.max(data, axis=0) <= 1)
    assert np.all(np.min(data, axis=0) >= 0)

    return data * (xmax - xmin) + xmin


def shuffle_data(data):
    """Randomly permutate the input."""
    try:
        import pandas as pd
        assert isinstance(data, (np.ndarray, pd.DataFrame))

        if isinstance(data, np.ndarray):
            n = len(data)
            shuffled_indices = np.random.permutation(n)
            return data[shuffled_indices]
        else:
            n = len(data.index)
            shuffled_indices = np.random.permutation(n)
            return data.iloc[shuffled_indices]
    except ImportError:
        logger.warning('Could not import Pandas.')


def swap_image_channel(np_arr):
    """Swap axes between 2nd and 4th for 4D inputs, or swap 1st and 3rd for 3D.
    """
    n = len(np_arr.shape)
    if n == 4:
        return np.swapaxes(np_arr, 1, 3)
    elif n == 3:
        return np.swapaxes(np_arr, 0, 2)
    else:
        return np_arr  # not a image, do nothing


def name_handler(filename, extension, overwrite=False):
    """Return a new filename based on current existence and extension.
    """
    arr = filename.split('.')

    if (len(arr) > 1 and arr[-1] != extension) or len(arr) == 1:
        arr.append(extension)
    filename = '.'.join(arr)

    # handle existing file
    if not overwrite and os.path.exists(filename):
        arr = filename.split('.')
        time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        arr.insert(-1, time_str)  # already fixed extension
        print('File {:s} already exists. Save new file as "{:s}"'.format(
            filename, '.'.join(arr)))

    return '.'.join(arr)


def onehot_encoding(y, num_classes, dtype=np.long):
    """Apply one hot encoding for given labels"""
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1  # should be an 1D array

    onehot = np.zeros((len(y), num_classes)).astype(dtype)
    onehot[np.arange(len(y)), y] = 1
    return onehot


def get_data_path():
    """Get absolute path for the `data` folder.
    """
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    logger.debug(path)
    return path


def get_l2_norm(a, b):
    """Computes the L2 norm between 2 samples"""
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert a.shape == b.shape
    assert len(a.shape) in (2, 4)

    n = len(a)
    l2 = np.sum(np.square(a.reshape(n, -1) - b.reshape(n, -1)), axis=1)
    return np.sqrt(l2)