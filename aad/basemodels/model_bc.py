"""
This module implements the PyTorch neural network model for Breast Cancer Wisconsin.
"""
import torch
import torch.nn as nn

LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam
OPTIM_PARAMS = {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01}
SCHEDULER = None
SCHEDULER_PARAMS = None
NUM_FEATURES = 30
NUM_CLASSES = 2


class BCNN(nn.Module):
    """Breast Cancer Neural Network
    Fully connected neural network tested on Breast Cancer Wisconsin.  
    This basic model should work on numeric binary classification
    """

    def __init__(
            self,
            num_features=NUM_FEATURES,
            num_classes=NUM_CLASSES,
            loss_fn=LOSS_FN,
            optimizer=OPTIMIZER,
            optim_params=OPTIM_PARAMS,
            scheduler=SCHEDULER,
            scheduler_params=SCHEDULER_PARAMS,
            from_logits=True):
        super(BCNN, self).__init__()

        self.hidden_model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64, num_classes)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.from_logits = from_logits

    def forward(self, x):
        x = self.hidden_model(x)
        x = self.fc1(x)
        if not self.from_logits:
            x = torch.softmax(x, dim=1)
        return x
