import time

import numpy as np
import torch
from art.attacks import SaliencyMapMethod
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer


class JacobianSaliencyContainer(AttackContainer):
    def __init__(self, model_container):
        super(JacobianSaliencyContainer, self).__init__(model_container)

        # TODO: implement this
        raise NotImplementedError

    def generate(self):
    # TODO: implement this
        raise NotImplementedError