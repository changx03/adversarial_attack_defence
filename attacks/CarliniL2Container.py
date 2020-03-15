import time

import numpy as np
import torch
from art.attacks import CarliniL2Method
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer


class CarliniL2Container(AttackContainer):
    def __init__(self, model_container):
        super(CarliniL2Container, self).__init__(model_container)

        # TODO: implement this
        raise NotImplementedError

    def generate(self):
        # TODO: implement this
        raise NotImplementedError
