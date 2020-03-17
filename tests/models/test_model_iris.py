import logging
import os
import unittest

import numpy as np
import torch

from aad.basemodels import IrisNN, TorchModelContainer
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, master_seed, swap_image_channel

logger = logging.getLogger(__name__)
SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 128


class TestModelIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        if not os.path.exists(os.path.join('save', 'test')):
            os.makedirs(os.path.join('save', 'test'))

        NAME = 'Iris'
        logger.info(f'Starting {NAME} data container...')
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=True)

        model = IrisNN()
        model_name = model.__class__.__name__
        logger.info(f'Using model: {model_name}')
        cls.mc = TorchModelContainer(model, cls.dc)
        cls.mc.fit(epochs=100, batch_size=BATCH_SIZE)

        # for comparison
        model2 = IrisNN()
        cls.mc2 = TorchModelContainer(model2, cls.dc)

        # inputs for testing
        cls.x = np.copy(cls.dc.data_test_np[:5])
        cls.y = np.copy(cls.dc.label_test_np[:5])

    def setUp(self):
        master_seed(SEED)

    def test_train(self):
        acc0 = self.mc.accuracy_test[-1]
        logger.debug('Test acc: {:.4f}'.format(acc0))

        # continue training
        self.mc.fit(epochs=50, batch_size=BATCH_SIZE)
        acc1 = self.mc.accuracy_test[-1]
        self.assertGreaterEqual(acc1, acc0)
        logger.debug('Test acc: {:.4f}'.format(acc1))

    def test_predict(self):
        x = torch.tensor(swap_image_channel(self.x)).to(self.mc.device)
        p1 = self.mc.predict(x)

        p2 = self.mc.predict(self.x)
        self.assertTrue((p1 == p2).all())

    def test_predict_one(self):
        x = torch.tensor(swap_image_channel(self.x[0])).to(self.mc.device)
        p1 = self.mc.predict_one(x)

        p2 = self.mc.predict_one(self.x[0])
        self.assertTrue((p1 == p2).all())

    def test_evaluate(self):
        p2 = self.mc.predict(self.x)
        acc = self.mc.evaluate(self.x, [1, 0, 1, 2, 2])
        self.assertEqual(acc, 1.0)


if __name__ == '__main__':
    unittest.main()
