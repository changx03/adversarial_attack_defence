import logging
import os
import unittest

from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import master_seed
from tests.utils import get_data_path
from aad.basemodels import MnistCnnCW, TorchModelContainer

logger = logging.getLogger(__name__)
SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 128


class TestModelMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NAME = 'MNIST'
        logger.info(f'Starting {NAME} data container...')
        dc = DataContainer(DATASET_LIST[NAME], get_data_path())

        model = MnistCnnCW()
        logger.info(f'Training {model.__name__}...')
        cls.mc = TorchModelContainer(model, dc)
        cls.mc.fit(epochs=5, batch_size=BATCH_SIZE)
        logger.info('Test acc: {:.4f}'.format(cls.mc.accuracy_test))

    def setUp(self):
        master_seed(SEED)

    def test_keep_training(self):
        acc0 = self.mc.accuracy_test

        self.mc.fit(epochs=5, batch_size=BATCH_SIZE)
        acc1 = self.mc.accuracy_test
        self.assertGreaterEqual(acc1, acc0)


if __name__ == '__main__':
    unittest.main()
