import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from ..basemodels import ModelContainerPT
from ..datasets import DATASET_LIST, DataContainer
from ..utils import get_data_path, is_probability
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class AdversarialTraining(DetectorContainer):
    def __init__(self,
                 model_container):
        super(AdversarialTraining, self).__init__(model_container)

    def fit(self):
        pass

    def save(self, filename, overwrite=False):
        pass

    def load(self, filename):
        pass

    def detect(self, adv, pred=None):
        pass
