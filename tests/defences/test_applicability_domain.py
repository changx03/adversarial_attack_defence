import logging
import unittest

import numpy as np
import torch

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
            cls.mc, hidden_model=hidden_model, k1=4, k2=6, confidence=0.8)

    def setUp(self):
        master_seed(SEED)

    def test_fit(self):
        result = self.ad.fit()
        self.assertTrue(result)
        t = self.ad.thresholds
        self.assertTrue((t != 0).all())
        # TODO: all thresholds has save value?!

        # test defence with testset
        adv = self.dc.data_test_np
        x_passed, blocked_indices = self.ad.defence(adv)
        print('Blocked {} inputs'.format(len(blocked_indices)))


if __name__ == '__main__':
    unittest.main()
