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
SAMPLE_RATIO = 1000 / 6e4


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

        hidden_model = model.hidden_model

        logger.info('sample_ratio: %f', SAMPLE_RATIO)
        cls.ad = ApplicabilityDomainContainer(
            cls.mc,
            hidden_model=hidden_model,
            k1=9,
            reliability=0.8,
            sample_ratio=SAMPLE_RATIO,
            confidence=0.9,
            kappa=10,
        )
        cls.ad.fit()

    def setUp(self):
        master_seed(SEED)

    def preform_attack(self, attack, count=100):
        adv, y_adv, x_clean, y_clean = attack.generate(count=count)

        accuracy = self.mc.evaluate(adv, y_clean)
        print('Accuracy on adversarial examples: {:.4f}%'.format(
            accuracy*100))

        x_passed, blocked_indices = self.ad.detect(adv)
        print('Blocked {:2d}/{:2d} samples from adversarial examples'.format(
            len(blocked_indices), len(adv)))
        print('blocked_indices', blocked_indices)

        matched_indices = np.where(y_adv == y_clean)[0]
        print('matched_indices', matched_indices)

        passed_indices = np.delete(np.arange(len(adv)), blocked_indices)
        passed_y_clean = y_clean[passed_indices]
        accuracy = self.mc.evaluate(x_passed, passed_y_clean)
        print('Accuracy on passed adversarial examples: {:.4f}%'.format(
            accuracy*100))
        return blocked_indices

    def test_fit_clean_small(self):
        x = self.dc.data_test_np[:5]
        x_passed, blocked_indices = self.ad.detect(x)
        print(f'# of blocked: {len(blocked_indices)}')
        self.assertGreaterEqual(len(x_passed)/len(x), 0.8)

    def test_fit_clean(self):
        x = self.dc.data_test_np
        x_passed, blocked_indices = self.ad.detect(x)
        print('Blocked {:2d}/{:2d} samples from clean samples'.format(
            len(blocked_indices), len(x)))
        self.assertGreaterEqual(len(x_passed)/len(x), 0.95)

    def test_fgsm_attack(self):
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True)
        blocked_indices = self.preform_attack(attack, count=NUM_ADV)
        blocked_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(blocked_rate, 0.35)


if __name__ == '__main__':
    unittest.main()
