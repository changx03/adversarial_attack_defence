"""
This module implements the data container from external source.
"""
import logging

from ..utils import get_range, scale_normalize
from .data_container import DataContainer

logger = logging.getLogger(__name__)


class CustomDataContainer(DataContainer):
    def __init__(self,
                 data_train,
                 label_train,
                 data_test,
                 label_test,
                 name,
                 data_type,
                 num_classes,
                 dim_data):
        """
        Create a CustomDataContainer instance

        Parameters
        ----------
        data_train : numpy.ndarray
            Input data for training.
        label_train : numpy.ndarray
            The labels of train data.
        data_test : numpy.ndarray
            Input data for evaluation.
        label_test : numpy.ndarray
        name : numpy.ndarray
            The labels of test data.
        data_type : 'image', 'numeric'
            The type of the data.
        num_classes : int
            Number of classes.
        dim_data : tuple
            The dimension of the data.
        """
        dataset_dict = {
            'name': name,
            'type': data_type,
            'num_classes': num_classes,
            'dim_data': dim_data,
        }
        if data_type not in ('image', 'numeric'):
            raise ValueError('Unkown data type.')
        super(CustomDataContainer, self).__init__(dataset_dict)
        self._x_train_np = data_train
        self._y_train_np = label_train
        self._x_test_np = data_test
        self._y_test_np = label_test

    def __call__(self, normalize=False):
        """
        Prepare data.

        Parameters
        ----------
        normalize : bool
            Apply normalization.
        """
        self._data_range = get_range(
            self._x_train_np, is_image=(self._data_type == 'image'))
        self._train_mean = self._x_train_np.mean(axis=0)
        self._train_std = self._x_train_np.std(axis=0)

        if normalize:
            (xmin, xmax) = self._data_range
            # NOTE: Carlini attack expects the data in range [0, 1]
            # mean = self._train_mean
            self._x_train_np = scale_normalize(
                self._x_train_np, xmin, xmax, mean=None)
            self._x_test_np = scale_normalize(
                self._x_test_np, xmin, xmax, mean=None)
