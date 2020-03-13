import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: You must call this function!
LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam
OPTIM_PARAMS = {'lr': 1e-4, 'betas': (0.9, 0.999)}
SCHEDULER = None
SCHEDULER_PARAMS = None


class BCNN(nn.Module):
    '''Breast Cancer Neural Network
    Fully connected neural network for Breast Cancer Wisconsin
    '''

    def __init__(
            self,
            loss_fn=LOSS_FN,
            optimizer=OPTIMIZER,
            optim_params=OPTIM_PARAMS,
            scheduler=SCHEDULER,
            scheduler_params=SCHEDULER_PARAMS):
        super(BCNN, self).__init__()

        self.model_inner = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = self.model_inner(x)
        return x
