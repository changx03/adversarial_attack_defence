import logging
import os
import unittest

import numpy as np
import torch

from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import FeatureSqueezing
from aad.utils import (get_data_path, get_pt_model_filename, get_range,
                       master_seed)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
# NAME = 'MNIST'
NAME = 'Iris'
MAX_EPOCHS = 10
# MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')
MODEL_FILE = os.path.join('save', 'IrisNN_Iris_e200.pt')


class TestFeatureSqueezing(unittest.TestCase):
    """Testing Feature Squeezing as Defence."""

    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        # Model = get_model('MnistCnnV2')
        Model = get_model('IrisNN')
        model = Model()
        logger.info('Starting %s data container...', NAME)
        dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        # dc()
        dc(normalize=True)
        cls.mc = ModelContainerPT(model, dc)
        cls.mc.load(MODEL_FILE)
        accuracy = cls.mc.evaluate(dc.x_test, dc.y_test)
        logger.info('Accuracy on test set: %f', accuracy)

    def setUp(self):
        master_seed(SEED)

    def test_fit_save(self):
        x_range = get_range(self.mc.data_container.x_train)
        squeezer = FeatureSqueezing(
            self.mc,
            clip_values=x_range,
            smoothing_methods=['median', 'normal', 'binary'],
            bit_depth=8,
            sigma=0.1,
            pretrained=False
        )
        squeezer.fit()

    def test_load(self):
        pass

    def test_clean_set(self):
        pass

    def test_detect(self):
        pass
