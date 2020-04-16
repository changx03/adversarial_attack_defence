"""
This module contains all static utility functions
"""
import copy
import logging
import math
import os
import time

import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)


def master_seed(seed):
    """
    Set the seed for all random number generators used in the library. This 
    ensures experiments reproducibility and stable testing.

    Parameters
    ----------
    seed : int
        The value to be seeded in the random number generators.
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
    """Returns (min, max) of a numpy array."""
    assert type(data) == np.ndarray

    axis = None if is_image else 0
    x_max = np.max(data, axis=axis)
    x_min = np.min(data, axis=axis)
    return (x_min, x_max)


def scale_normalize(data, xmin, xmax, mean=None):
    """
    Applies scaling. If mean is not none, set output to zero mean. Otherwise,
    it scales data to [0, 1]
    """
    assert isinstance(data, np.ndarray) \
        and isinstance(xmin, (np.ndarray, np.float32, float)) \
        and isinstance(xmax, (np.ndarray, np.float32, float)) \
        and isinstance(mean, (type(None), np.ndarray, np.float32, float))

    # handle image data
    if len(data.shape) == 4:  # (n,h,w,c)
        mean = np.mean(mean) if mean is not None else None
        xmin = np.min(xmin)
        xmax = np.max(xmax)

    if mean is not None:
        return (data - mean) / (xmax - xmin)
    return (data - xmin) / (xmax - xmin)


def scale_unnormalize(data, xmin, xmax, mean=None):
    """Rescales the normalized data back to raw."""
    assert isinstance(data, np.ndarray) \
        and isinstance(xmin, np.ndarray) \
        and isinstance(xmax, np.ndarray) \
        and isinstance(mean, (type(None), np.ndarray))
    assert data.shape[1] == len(xmax) \
        and data.shape[1] == len(xmin)

    if mean is not None:
        assert data.shape[1] == len(mean)
        return data * (xmax - xmin) + mean
    return data * (xmax - xmin) + xmin


def shuffle_data(data):
    """Randomly permutate the input."""
    try:
        import pandas as pd
        assert isinstance(data, (np.ndarray, pd.DataFrame))

        if isinstance(data, np.ndarray):
            return np.random.permutation(data)
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


def name_handler(filename, extension, overwrite):
    """Return a new filename based on current existence and extension.
    """
    arr = filename.split('.')

    if (len(arr) > 1 and arr[-1] != extension) or len(arr) == 1:
        arr.append(extension)
    new_name = '.'.join(arr)

    # handle adv name postfix
    temp_name = new_name.replace('[ph]', 'adv')

    # handle existing file
    if not overwrite and os.path.exists(temp_name):
        arr = new_name.split('.')
        time_str = get_time_str()
        arr[-2] += '_' + time_str  # already fixed extension
        new_name = '.'.join(arr)
        logger.info('File %s already exists. Save new file as %s',
                    filename, new_name)
    return new_name


def onehot_encoding(y, num_classes, dtype=np.int64):
    """Apply one hot encoding for given labels"""
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1  # should be an 1D array

    onehot = np.zeros((len(y), num_classes)).astype(dtype)
    onehot[np.arange(len(y)), y] = 1
    return onehot


def get_data_path():
    """Get absolute path for the `data` folder."""
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


def get_time_str():
    """Returns the formated local time in [YearMonthDateHourMinSec]"""
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def get_pt_model_filename(model_name, dataset, epochs):
    """Return the filename for PyTorch model"""
    return '{}_{}_e{}.pt'.format(model_name, dataset, epochs)


def is_probability(vector):
    """Check if the score add up to 1."""
    assert isinstance(vector, np.ndarray) and len(vector.shape) == 1
    sum_to_1 = math.isclose(vector.sum(), 1.0, rel_tol=1e-3)
    smaller_than_1 = np.amax(vector) <= 1.0
    larger_than_0 = np.all(vector >= 0.0)
    return sum_to_1 and smaller_than_1 and larger_than_0


def copy_model(model, pretrained=False):
    """Create a copy of a model"""
    assert isinstance(model, nn.Module)
    Model = model.__class__
    loss_fn = model.loss_fn
    optimizer = model.optimizer
    optim_params = model.optim_params
    scheduler = model.scheduler
    scheduler_params = model.scheduler_params
    from_logits = model.from_logits
    clone = Model(
        loss_fn=loss_fn,
        optimizer=optimizer,
        optim_params=optim_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        from_logits=from_logits,
    )
    if pretrained:
        state_dict = copy.deepcopy(model.state_dict())
        clone.load_state_dict(state_dict)

    return clone


def get_random_targets(true_labels,
                       num_classes,
                       use_onehot=False,
                       dtype=np.int64):
    """
    Get randomly generated targets exclude the true labels

    Parameters
    ----------
    true_labels : numpy.ndarray
        List of true labels.
    num_classes : int
        Number of classes.
    use_onehot : bool
        Returns one-hot encoded targets, if it is true.
    dtype : type
        The data type of returned value. Default = np.int64
    Returns
    -------
    numpy.ndarray
        The target labels. It has same shape as `true_labels`.
    """
    count = len(true_labels)
    classes = np.arange(num_classes)
    targets = -np.ones_like(true_labels, dtype=np.int64)
    for i in range(count):
        targets[i] = np.random.choice(np.delete(classes, true_labels[i]))
    if not use_onehot:
        return targets
    else:
        return onehot_encoding(targets, num_classes, dtype)
