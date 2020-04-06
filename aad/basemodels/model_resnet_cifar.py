"""
This module implements fine-tune a pretrained ResNet50 for CIFAR10
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = optim.SGD
OPTIM_PARAMS = {'lr': 0.001, 'momentum': 0.9}
SCHEDULER = optim.lr_scheduler.StepLR
SCHEDULER_PARAMS = {'step_size': 10, 'gamma': 0.5}


class CifarResnet50(nn.Module):
    """
    A fine-tune model for CIFAR10 dataset. It uses pretrained ResNet-50 from PyTorch. 
    """

    def __init__(
            self,
            loss_fn=LOSS_FN,
            optimizer=OPTIMIZER,
            optim_params=OPTIM_PARAMS,
            scheduler=SCHEDULER,
            scheduler_params=SCHEDULER_PARAMS):
        super(CifarResnet50, self).__init__()

        resnet = tv.models.resnet50(pretrained=True, progress=False)
        self.resnet = resnet
        num_fcin = resnet.fc.in_features
        resnet.fc = nn.Linear(num_fcin, 10)
        self.hidden_model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()
        )

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x):
        return self.resnet(x)
