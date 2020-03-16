import torch


class NumeralDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        assert isinstance(data, torch.Tensor) \
            and isinstance(label, torch.Tensor)

        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
