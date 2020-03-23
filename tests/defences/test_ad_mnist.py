import logging
import os
import unittest

import numpy as np

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         FGSMContainer, SaliencyContainer, ZooContainer)
from aad.basemodels import MnistCnnCW, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, master_seed

logger = logging.getLogger(__name__)

SEED = 4096
BATCH_SIZE = 128
NUM_ADV = 100  # number of adversarial examples will be generated
NAME = 'MNIST'
FILE_NAME = 'example-mnist-e20.pt'


class TestApplicabilityDomainMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=False)

        model = MnistCnnCW()
        logger.info('Using model: %s', model.__class__.__name__)

        cls.mc = ModelContainerPT(model, cls.dc)

        file_path = os.path.join('save', FILE_NAME)
        if not os.path.exists(file_path):
            cls.mc.fit(epochs=20, batch_size=BATCH_SIZE)
            cls.mc.save(FILE_NAME, overwrite=True)
        else:
            logger.info('Use saved parameters from %s', FILE_NAME)
            cls.mc.load(file_path)

        accuracy = cls.mc.evaluate(cls.dc.data_test_np, cls.dc.label_test_np)
        logger.info('Accuracy on test set: %f', accuracy)

        cls.hidden_model = model.hidden_model

    def setUp(self):
        master_seed(SEED)

    def test_search_params(self):
        # best k2 = 24
        min_blocked = np.inf
        best_k2 = -1
        lookup_list = np.arange(0, 31) * 2
        lookup_list[0] = 1
        logger.debug(str(lookup_list))
        for k in lookup_list:
            logger.debug('Current k = %i', k)
            ad = ApplicabilityDomainContainer(
                self.mc,
                hidden_model=self.hidden_model,
                k1=4,
                k2=k,
                confidence=1.8)
            ad.fit()

            x = self.dc.data_test_np
            x_passed, blocked_indices = ad.detect(x)
            num_blocked = len(blocked_indices)
            if min_blocked > num_blocked:
                best_k2 = k
                min_blocked = num_blocked
        print(f'Best k2 = {best_k2}, Blocked: {min_blocked}')


if __name__ == '__main__':
    unittest.main()
