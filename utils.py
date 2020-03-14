import datetime
import os

import numpy as np
import pandas as pd


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


def scale_unnormalize(data, xmin, xmax):
    '''rescaling the normalized data back to raw
    '''
    assert (type(data) == np.ndarray and
            type(xmax) == np.ndarray and
            type(xmin) == np.ndarray)
    assert data.shape[1] == len(xmax) and data.shape[1] == len(xmin)
    assert np.all(np.max(data, axis=0) <= 1)
    assert np.all(np.min(data, axis=0) >= 0)

    return data * (xmax - xmin) + xmin


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


def swap_image_channel(np_arr):
    n = len(np_arr.shape)
    if n == 4:
        return np.swapaxes(np_arr, 1, 3)
    elif n == 3:
        return np.swapaxes(np_arr, 0, 2)
    else:
        raise Exception('Not enough axes for swapping!')


def name_handler(filename, extension='pt', overwrite=False):
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