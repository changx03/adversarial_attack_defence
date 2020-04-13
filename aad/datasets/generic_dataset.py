"""
This module implements a generic Dataset. It can be used for both Tensor and numpy array.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    """
    Class implements a generic Dataset. It can be used for both Tensor and numpy array.
    """

    def __init__(self, data, labels=None):
        """
        Create a Dataset instance.

        Parameters
        ----------
        data : torch.Tensor, numpy.ndarray
            Input data.
        labels : torch.Tensor, numpy.ndarray, optional
            Input labels.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)) \
            and isinstance(labels, (torch.Tensor, np.ndarray, type(None)))

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        self.data = data
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index] if isinstance(
            self.labels, torch.Tensor) else -1
        return self.data[index], label

    def __len__(self):
        return len(self.data)
