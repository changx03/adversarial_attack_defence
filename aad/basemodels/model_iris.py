import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: You must call this function!
LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.SGD
OPTIM_PARAMS = {'lr': 0.01, 'momentum': 0.9}
SCHEDULER = None
SCHEDULER_PARAMS = None
NUM_FEATURES = 4
NUM_CLASSES = 3
HIDDEN_NODES = 16


class IrisNN(nn.Module):
    """Iris Neural Network
    Fully connected neural network tested on Iris.  
    It has 2 hidden layers, the number of hidden nodes per layer can be adjusted.
    """

    def __init__(
            self,
            num_features=NUM_FEATURES,
            num_classes=NUM_CLASSES,
            hidden_nodes=HIDDEN_NODES,
            loss_fn=LOSS_FN,
            optimizer=OPTIMIZER,
            optim_params=OPTIM_PARAMS,
            scheduler=SCHEDULER,
            scheduler_params=SCHEDULER_PARAMS):
        super(IrisNN, self).__init__()

        self.hidden_model = nn.Sequential(
            nn.Linear(num_features, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU())
        self.fc1 = nn.Linear(hidden_nodes, num_classes)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = self.hidden_model(x)
        x = self.fc1(x)
        return x
