import torch


class NumeralDataset(torch.utils.data.Dataset):
    def __init__(self, data, label=None):
        assert isinstance(data, torch.Tensor) \
            and type(label) in (torch.Tensor, None)

        self.data = data
        self.label = label

    def __getitem__(self, index):
        y = self.label[index] if self.label else -1
        return self.data[index], y

    def __len__(self):
        return len(self.data)
