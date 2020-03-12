import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCnn1_Encoder(nn.Module):
    def __init__(self):
        super(MnistCnn1_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MnistCnn1(nn.Module):
    def __init__(self, lr=0.1, momentum=0.9):
        super(MnistCnn1, self).__init__()
        self.lr = lr

        self.model_hidden = MnistCnn1_Encoder()
        self.model_no_softmax = nn.Sequential(
            self.model_hidden,
            torch.nn.Linear(128, 10))

        self.optimizer = torch.optim.SGD
        self.scheduler = None

    def forward(self, x):
        x = self.model_no_softmax(x)
        x = F.log_softmax(x, dim=1)
        return x


# for testing only
# if __name__ == '__main__':
#     x = torch.rand(2, 1, 28, 28)
#     model = MnistCnn1()
#     y = model(x)
#     print(y)

#     print(isinstance(model.model_hidden, nn.Module))
#     print(isinstance(model.model_no_softmax, nn.Module))
#     print(isinstance(model, nn.Module))
#     print(model.optimizer)
