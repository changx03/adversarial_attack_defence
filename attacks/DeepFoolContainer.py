import time

import numpy as np
import torch
from art.attacks import DeepFool
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer


class DeepFoolContainer(AttackContainer):
    def __init__(self, model_container):
        super(DeepFoolContainer, self).__init__(model_container)

        # TODO: implement this
        raise NotImplementedError

    def generate(self):
        # TODO: implement this
        raise NotImplementedError