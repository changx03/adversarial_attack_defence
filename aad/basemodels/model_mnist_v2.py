"""
This module implements the PyTorch neural network model for MNIST.
Two changes have been made:
1. It uses Adam optimizer instead of SGD
2. The hidden layer is on the layer before output layer, instead of the layer
   after convolutional layer.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam
OPTIM_PARAMS = {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.001}
SCHEDULER = None
SCHEDULER_PARAMS = None


class MnistCnnV2(nn.Module):
    """
    A convolutional neural network for MNIST. The same structure was used in
    Carlini and Wagner attack
    """

    def __init__(
            self,
            loss_fn=LOSS_FN,
            optimizer=OPTIMIZER,
            optim_params=OPTIM_PARAMS,
            scheduler=SCHEDULER,
            scheduler_params=SCHEDULER_PARAMS):
        super(MnistCnnV2, self).__init__()

        self.hidden_model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),
            ('flat1', nn.Flatten()),
            ('fc1', nn.Linear(1024, 200)),
            ('relu5', nn.ReLU()),
            ('fc2', nn.Linear(200, 128)),
            ('relu6', nn.ReLU()),
        ]))
        self.fn = nn.Linear(128, 10)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = self.hidden_model(x)
        x = self.fn(x)
        return x
