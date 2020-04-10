import logging
import os
import unittest

import numpy as np
import torch

from aad.basemodels import MnistCnnCW, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, master_seed, swap_image_channel

logger = logging.getLogger(__name__)
SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 128


class TestModelMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        if not os.path.exists(os.path.join('save', 'test')):
            os.makedirs(os.path.join('save', 'test'))

        NAME = 'MNIST'
        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=False)

        model = MnistCnnCW()
        model_name = model.__class__.__name__
        logger.info('Using model: %s', model_name)
        cls.mc = ModelContainerPT(model, cls.dc)
        cls.mc.fit(max_epochs=2, batch_size=BATCH_SIZE)

        # for comparison
        model2 = MnistCnnCW()
        cls.mc2 = ModelContainerPT(model2, cls.dc)

        # inputs for testing
        cls.x = np.copy(cls.dc.data_test_np[:5])
        cls.y = np.copy(cls.dc.label_test_np[:5])

    def setUp(self):
        master_seed(SEED)

    def test_train(self):
        self.mc.fit(max_epochs=2, batch_size=BATCH_SIZE)
        acc0 = self.mc.accuracy_test[-1]
        logger.debug('Test acc: %f', acc0)

        # continue training
        self.mc.fit(max_epochs=2, batch_size=BATCH_SIZE)
        acc1 = self.mc.accuracy_test[-1]
        self.assertGreaterEqual(acc1, acc0)
        logger.debug('Test acc: %f', acc1)

    def test_save(self):
        self.mc.save(os.path.join('test', 'test_mnist'), overwrite=True)
        full_path = os.path.join('save', 'test', 'test_mnist.pt')
        self.assertTrue(os.path.exists(full_path))

    def test_load(self):
        self.mc.save(os.path.join('test', 'test_mnist'), overwrite=True)
        full_path = os.path.join('save', 'test', 'test_mnist.pt')
        self.mc2.load(full_path)

        x = torch.tensor(swap_image_channel(self.x)).to(self.mc.device)
        s1 = self.mc.model(x).cpu().detach().numpy()
        s2 = self.mc2.model(x).cpu().detach().numpy()
        np.testing.assert_almost_equal(s1, s2)

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
        acc = self.mc.evaluate(self.x, [7, 2, 1, 0, 4])
        self.assertEqual(acc, 1.0)


if __name__ == '__main__':
    unittest.main()
