"""
This module implements the PyTorch neural network model for MNIST.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam
OPTIM_PARAMS = {'lr':0.001, 'betas':(0.9, 0.999), 'weight_decay': 0.001}
SCHEDULER = None
SCHEDULER_PARAMS = None


class CifarCnn(nn.Module):
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
        super(CifarCnn, self).__init__()

        self.hidden_model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(64, 128, 3)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(128, 128, 3)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),
            ('flat1', nn.Flatten()),
            ('fc1', nn.Linear(3200, 512)),
            ('relu5', nn.ReLU()),
            ('fc2', nn.Linear(512, 512)),
            ('relu6', nn.ReLU()),
        ]))
        self.fn = nn.Linear(512, 10)
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = self.hidden_model(x)
        x = self.fn(x)
        return x
