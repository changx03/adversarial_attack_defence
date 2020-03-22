import logging
import unittest

import numpy as np
import torch

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         FGSMContainer, ZooContainer)
from aad.basemodels import IrisNN, ModelContainerPT
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

        cls.num_adv = 30  # number of adversarial examples will be generated

        NAME = 'Iris'
        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=True, normalize=False)

        model = IrisNN(hidden_nodes=10)
        model_name = model.__class__.__name__
        logger.info('Using model: %s', model_name)
        cls.mc = ModelContainerPT(model, cls.dc)
        cls.mc.fit(epochs=120, batch_size=BATCH_SIZE)

        hidden_model = model.hidden_model
        cls.ad = ApplicabilityDomainContainer(
            cls.mc, hidden_model=hidden_model, k1=3, k2=7, confidence=.8)
        cls.ad.fit()

    def setUp(self):
        master_seed(SEED)

    def preform_attack(self, attack, count=30):
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

    def test_fit_clean(self):
        """
        Testing defence against clean inputs (false positive case).
        """
        x = self.dc.data_test_np
        print('Number of test samples: {}'.format(len(x)))
        y = self.dc.label_test_np
        x_passed, blocked_indices = self.ad.detect(x)
        self.assertEqual(len(x_passed) + len(blocked_indices), len(x))
        print('Blocked {:2d}/{:2d} samples from clean samples'.format(
            len(blocked_indices), len(x)))
        print(blocked_indices)
        passed_indices = np.delete(np.arange(len(x)), blocked_indices)
        passed_y_clean = y[passed_indices]
        accuracy = self.mc.evaluate(x_passed, passed_y_clean)
        print('Accuracy on clean samples: {:.4f}%'.format(accuracy*100))

    def test_fgsm_attack(self):
        """
        Testing defence against FGSM attack.
        NOTE: The block rate is low due to low success rate for the attack
        """
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True)
        blocked_indices = self.preform_attack(attack, count=self.num_adv)
        blocked_rate = len(blocked_indices) / self.num_adv
        self.assertGreater(blocked_rate, 0.35)

    def test_bim_attack(self):
        """
        Testing defence against BIM attack.
        """
        attack = BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False)
        blocked_indices = self.preform_attack(attack, count=self.num_adv)
        blocked_rate = len(blocked_indices) / self.num_adv
        self.assertGreater(blocked_rate, 0.5)

    def test_deepfool_attack(self):
        """
        Testing defence against DeepFool attack.
        """
        attack = DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10)
        blocked_indices = self.preform_attack(attack, count=self.num_adv)
        blocked_rate = len(blocked_indices) / self.num_adv
        self.assertGreater(blocked_rate, 0.5)

    def test_carlini_l2_attack(self):
        """
        Testing defence against Carlini & Wagner L2 attack
        """
        attack = CarliniL2Container(
            self.mc,
            confidence=0.0,
            targeted=False,
            learning_rate=1e-2,
            binary_search_steps=10,
            max_iter=100,
            initial_const=1e-2,
            max_halving=5,
            max_doubling=10,
            batch_size=8)
        blocked_indices = self.preform_attack(attack, count=self.num_adv)
        blocked_rate = len(blocked_indices) / self.num_adv
        self.assertGreater(blocked_rate, 0.5)


if __name__ == '__main__':
    unittest.main()
