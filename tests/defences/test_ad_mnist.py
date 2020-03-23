import logging
import unittest
import os

import numpy as np

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         FGSMContainer, JacobianSaliencyContainer, ZooContainer)
from aad.basemodels import MnistCnnCW, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, master_seed

logger = logging.getLogger(__name__)
SEED = 4096
BATCH_SIZE = 128
NUM_ADV = 100  # number of adversarial examples will be generated
NAME = 'MNIST'
FILE_NAME = 'example-mnist-e30.pt'


class TestApplicabilityDomainMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        logger.info('Starting {} data container...'.format(NAME))
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=False)

        model = MnistCnnCW()
        logger.info('Using model: {}'.format(model.__class__.__name__))

        cls.mc = ModelContainerPT(model, cls.dc)

        file_path = os.path.join('save', FILE_NAME)
        if not os.path.exists(file_path):
            cls.mc.fit(epochs=30, batch_size=BATCH_SIZE)
            cls.mc.save(FILE_NAME, overwrite=True)
        else:
            logger.info('Use saved parameters from {}'.format(FILE_NAME))
            cls.mc.load(file_path)

        acc = cls.mc.evaluate(cls.dc.data_test_np, cls.dc.label_test_np)
        logger.info('Accuracy on test set: {:.4f}%'.format(acc*100))

    def setUp(self):
        master_seed(SEED)

    def test_fit_clean(self):
        pass


if __name__ == '__main__':
    unittest.main()

