"""
This module implements a custom PyTorch Dataset for NumPy array
"""
import torch


class NumericalDataset(torch.utils.data.Dataset):
    def __init__(self, data, label=None):
        assert isinstance(data, torch.Tensor) \
            and isinstance(label, (torch.Tensor, type(None)))

        self.data = data
        self.label = label

    def __getitem__(self, index):
        y = self.label[index] if isinstance(self.label, torch.Tensor) else -1
        return self.data[index], y

    def __len__(self):
        return len(self.data)
