import logging
import os
import unittest

import numpy as np
import torch

from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import FeatureSqueezing
from aad.utils import get_data_path, get_pt_model_filename, master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
NAME = 'MNIST'
MAX_EPOCHS = 10


class TestFeatureSqueezing(unittest.TestCase):
    """Testing Feature Squeezing as Defence."""

    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

    def setUp(self):
        master_seed(SEED)

    def test_fit_save(self):
        pass

    def test_load(self):
        pass

    def test_clean_set(self):
        pass

    def test_detect(self):
        pass
