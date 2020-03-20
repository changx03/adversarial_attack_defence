import logging
import unittest

import numpy as np
import torch

from aad.attacks import BIMContainer
from aad.basemodels import IrisNN, TorchModelContainer
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, master_seed

logger = logging.getLogger(__name__)
SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 128


class TestApplicabilityDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        NAME = 'Iris'
        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=True, normalize=False)

        model = IrisNN(hidden_nodes=10)
        model_name = model.__class__.__name__
        logger.info('Using model: %s', model_name)
        cls.mc = TorchModelContainer(model, cls.dc)
        cls.mc.fit(epochs=120, batch_size=BATCH_SIZE)

        hidden_model = model.hidden_model
        cls.ad = ApplicabilityDomainContainer(
            cls.mc, hidden_model=hidden_model, k1=3, k2=7, confidence=.8)
        cls.ad.fit()

    def setUp(self):
        master_seed(SEED)

    def test_fit_clean(self):
        """
        testing defence with testset
        """
        adv = self.dc.data_test_np
        x_passed, blocked_indices = self.ad.defence(adv)
        self.assertEqual(len(x_passed) + len(blocked_indices), len(adv))
        self.assertTrue(np.equal(blocked_indices, [4, 7]).all())

    def test_adv(self):
        """
        testing defence with adversarial examples
        """
        attack = BIMContainer(self.mc)
        adv, y_adv, x_clean, y_clean = attack.generate(count=30)
        x_passed, blocked_indices = self.ad.defence(adv)
        self.assertEqual(len(x_passed) + len(blocked_indices), len(adv))
        print('Blocked {:2d}/{:2d} samples from adversarial examples'.format(
            len(blocked_indices), len(adv)))
        print(blocked_indices)
        passed_indices = np.delete(np.arange(len(adv)), blocked_indices)
        passed_y_clean = y_clean[passed_indices]
        accuracy = self.mc.evaluate(x_passed, passed_y_clean)
        print('Accuracy on passed adversarial examples: {:.4f}%'.format(
            accuracy*100))

        # test the base inputs
        print('\nTesting on original clean samples')
        x_passed, blocked_indices = self.ad.defence(x_clean)
        self.assertEqual(len(x_passed) + len(blocked_indices), len(adv))
        print('Blocked {:2d}/{:2d} samples from clean samples'.format(
            len(blocked_indices), len(adv)))
        print(blocked_indices)
        passed_indices = np.delete(np.arange(len(adv)), blocked_indices)
        passed_y_clean = y_clean[passed_indices]
        accuracy = self.mc.evaluate(x_passed, passed_y_clean)
        print('Accuracy on clean samples: {:.4f}%'.format(accuracy*100))


if __name__ == '__main__':
    unittest.main()
