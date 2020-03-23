import logging
import unittest
import os

import numpy as np

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         FGSMContainer, JacobianSaliencyContainer, ZooContainer)
from aad.basemodels import MnistCnnCW, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, master_seed, get_l2_norm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    def test_fgsm(self):
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True
        )
        print(attack)
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: {:.4f}%'.format(
            accuracy*100))

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())
        # Expect above 28% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.28)
        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        self.assertLessEqual(dif, 0.3)
        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)

    def test_bim(self):
        attack = BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False
        )
        print(attack)
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: {:.4f}%'.format(
            accuracy*100))

        self.assertFalse((adv == x_clean).all())
        # Expect above 90% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.9)
        # Check the max perturbation. The result is slightly higher than 0.3
        dif = np.max(np.abs(adv - x_clean))
        self.assertLessEqual(dif, 0.3 + 1e6)
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)

    def test_carlini(self):
        """
        Carlini and Wagner attack uses a greedy search method. The success rate
        can be 100%. However the algorithm try to match the target with minimal 
        perturbations, but there is no l2 threshold during generation.
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
            batch_size=8,
        )
        # Slow algorithm, only test 30 samples
        adv, y_adv, x_clean, y_clean = attack.generate(count=30)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: {:.4f}%'.format(
            accuracy*100))

        self.assertFalse((adv == x_clean).all())
        # Expect above 90% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.9)
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)

    def test_deepfool(self):
        attack = DeepFoolContainer(
            self.mc,
            max_iter=100, 
            epsilon=1e-6,
            nb_grads=10, 
            batch_size=16
        )
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adversarial examples: {:.4f}%'.format(
            accuracy*100))

        self.assertFalse((adv == x_clean).all())
        # Expect above 65% success rate
        self.assertGreaterEqual(
            (y_adv != y_clean).sum() / x_clean.shape[0], 0.65)
        self.assertLessEqual(np.max(adv), 1.0 + 1e6)
        self.assertGreaterEqual(np.min(adv), 0 - 1e6)
        l2 = get_l2_norm(adv, x_clean)
        print(l2)

    def test_saliency(self):
        pass

    def test_zoo(self):
        pass


if __name__ == '__main__':
    unittest.main()
