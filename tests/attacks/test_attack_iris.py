import logging
import os
import unittest

import numpy as np

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         FGSMContainer)
from aad.basemodels import IrisNN, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, get_l2_norm, master_seed

logger = logging.getLogger(__name__)

SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 256  # Train the entire set in one batch
NUM_ADV = 30  # number of adversarial examples will be generated
NAME = 'Iris'
FILE_NAME = 'example-iris-e200.pt'


class TestAttackIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        # ordered by labels, it requires shuffle!
        cls.dc(shuffle=True, normalize=False)

        model = IrisNN(hidden_nodes=12)
        logger.info('Using model: %s', model.__class__.__name__)

        cls.mc = ModelContainerPT(model, cls.dc)

        file_path = os.path.join('save', FILE_NAME)
        if not os.path.exists(file_path):
            cls.mc.fit(epochs=200, batch_size=BATCH_SIZE)
            cls.mc.save(FILE_NAME, overwrite=True)
        else:
            logger.info('Use saved parameters from %s', FILE_NAME)
            cls.mc.load(file_path)

        accuracy = cls.mc.evaluate(cls.dc.data_test_np, cls.dc.label_test_np)
        logger.info('Accuracy on test set: %f', accuracy)

    def setUp(self):
        master_seed(SEED)

    def test_fgsm(self):
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True)
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: %f', accuracy)

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())
        # Expect above 26% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.26)
        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        self.assertLessEqual(dif, 0.3)
        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)
        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_bim(self):
        attack = BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False)
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: %f', accuracy)

        self.assertFalse((adv == x_clean).all())
        # Expect above 46% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.46)
        # Check the max perturbation. The result is slightly higher than 0.3
        dif = np.max(np.abs(adv - x_clean))
        self.assertLessEqual(dif, 0.3 + 1e6)
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)
        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_carlini(self):
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
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: %f', accuracy)

        self.assertFalse((adv == x_clean).all())
        # Expect above 90% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.9)
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)
        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_deepfool(self):
        attack = DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10)
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: %f', accuracy)

        self.assertFalse((adv == x_clean).all())
        # Expect above 86% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.86)
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)
        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)


if __name__ == '__main__':
    unittest.main()
