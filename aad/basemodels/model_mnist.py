"""
This module implements the PyTorch neural network model for MNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_FN = nn.NLLLoss()
OPTIMIZER = torch.optim.SGD
OPTIM_PARAMS = {'lr': 0.1, 'momentum': 0}
SCHEDULER = None
SCHEDULER_PARAMS = None


class MnistCnnCW_hidden(nn.Module):
    def __init__(self):
        super(MnistCnnCW_hidden, self).__init__()
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


class MnistCnnCW(nn.Module):
    def __init__(
            self,
            loss_fn=LOSS_FN,
            optimizer=OPTIMIZER,
            optim_params=OPTIM_PARAMS,
            scheduler=SCHEDULER,
            scheduler_params=SCHEDULER_PARAMS):
        super(MnistCnnCW, self).__init__()

        self.model_hidden = MnistCnnCW_hidden()
        self.model_no_softmax = nn.Sequential(
            self.model_hidden,
            torch.nn.Linear(128, 10))

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = self.model_no_softmax(x)
        x = F.log_softmax(x, dim=1)
        return x
