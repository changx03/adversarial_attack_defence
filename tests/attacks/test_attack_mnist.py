import logging
import os
import unittest

import numpy as np

import aad.attacks as attacks
from aad.basemodels import MnistCnnCW, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, get_l2_norm, master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
NUM_ADV = 100  # number of adversarial examples will be generated
NAME = 'MNIST'
MAX_EPOCHS = 30
FILE_NAME = os.path.join('test', 'test-mnist-e30.pt')


class TestAttackMNIST(unittest.TestCase):
    """Test Adversarial Attacks on MNIST dataset"""

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
            cls.mc.fit(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
            cls.mc.save(FILE_NAME, overwrite=True)
        else:
            logger.info('Use saved parameters from %s', FILE_NAME)
            cls.mc.load(file_path)

        accuracy = cls.mc.evaluate(cls.dc.x_test, cls.dc.y_test)
        logger.info('Accuracy on test set: %f', accuracy)

    def setUp(self):
        master_seed(SEED)

    def test_fgsm(self):
        attack = attacks.FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True
        )
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())

        # test accuracy
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)
        self.assertLessEqual(accuracy, 0.53)

        # test success rate
        success_rate = (y_adv != y_clean).sum() / len(y_adv)
        logger.info('Success rate of adv. attack: %f', success_rate)
        self.assertGreaterEqual(success_rate, 0.47)

        # sum success rate (missclassified) and accuracy (correctly classified)
        self.assertAlmostEqual(success_rate + accuracy, 1.0, places=4)

        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        logger.info('Max perturbation (L1-norm): %f', dif)
        self.assertLessEqual(dif, 0.3 + 1e-4)

        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e-4)
        self.assertGreaterEqual(np.min(adv), 0 - 1e-4)

        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_bim(self):
        attack = attacks.BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False
        )
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())

        # test accuracy
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)
        self.assertLessEqual(accuracy, 0.1)

        # test success rate
        success_rate = (y_adv != y_clean).sum() / len(y_adv)
        logger.info('Success rate of adv. attack: %f', success_rate)
        self.assertGreaterEqual(success_rate, 0.9)

        # sum success rate (missclassified) and accuracy (correctly classified)
        self.assertAlmostEqual(success_rate + accuracy, 1.0, places=4)

        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        logger.info('Max perturbation (L1-norm): %f', dif)
        self.assertLessEqual(dif, 0.3 + 1e-4)

        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e-4)
        self.assertGreaterEqual(np.min(adv), 0 - 1e-4)

        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_carlini(self):
        attack = attacks.CarliniL2V2Container(
            self.mc,
            learning_rate=0.01,
            binary_search_steps=9,
            max_iter=1000,
            confidence=0.0,
            initial_const=0.01,
            c_range=(0, 1e10),
            batch_size=16,
            clip_values=(0.0, 1.0)
        )
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())

        # test accuracy
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)
        self.assertLessEqual(accuracy, 0.60)

        # test success rate
        success_rate = (y_adv != y_clean).sum() / len(y_adv)
        logger.info('Success rate of adv. attack: %f', success_rate)
        self.assertGreaterEqual(success_rate, 0.40)

        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        logger.info('Max perturbation (L1-norm): %f', dif)
        self.assertLessEqual(dif, 1.0 + 1e-4)

        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e-4)
        self.assertGreaterEqual(np.min(adv), 0 - 1e-4)

        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_deepfool(self):
        attack = attacks.DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10,
            batch_size=16
        )
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())

        # test accuracy
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)
        self.assertLessEqual(accuracy, 0.60)

        # test success rate
        success_rate = (y_adv != y_clean).sum() / len(y_adv)
        logger.info('Success rate of adv. attack: %f', success_rate)
        self.assertGreaterEqual(success_rate, 0.40)

        # sum success rate (missclassified) and accuracy (correctly classified)
        self.assertAlmostEqual(success_rate + accuracy, 1.0, places=4)

        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        logger.info('Max perturbation (L1-norm): %f', dif)
        self.assertLessEqual(dif, 1.0 + 1e-4)

        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e-4)
        self.assertGreaterEqual(np.min(adv), 0 - 1e-4)

        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_saliency(self):
        attack = attacks.SaliencyContainer(
            self.mc,
            theta=0.1,
            gamma=1.0,
            batch_size=16
        )
        adv, y_adv, x_clean, y_clean = attack.generate(count=NUM_ADV)

        # At least made some change from clean images
        self.assertFalse((adv == x_clean).all())

        # test accuracy
        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)
        self.assertLessEqual(accuracy, 0.1)

        # test success rate
        success_rate = (y_adv != y_clean).sum() / len(y_adv)
        logger.info('Success rate of adv. attack: %f', success_rate)
        self.assertGreaterEqual(success_rate, 0.9)

        # sum success rate (missclassified) and accuracy (correctly classified)
        self.assertAlmostEqual(success_rate + accuracy, 1.0, places=4)

        # Check the max perturbation
        dif = np.max(np.abs(adv - x_clean))
        logger.info('Max perturbation (L1-norm): %f', dif)
        self.assertLessEqual(dif, 1.0 + 1e-4)

        # Check bounding box
        self.assertLessEqual(np.max(adv), 1.0 + 1e-4)
        self.assertGreaterEqual(np.min(adv), 0 - 1e-4)

        l2 = np.max(get_l2_norm(adv, x_clean))
        logger.info('L2 norm = %f', l2)

    def test_zoo(self):
        pass
    #     """
    #     NOTE: This is a CPU only implementation. Extremely slow and has low
    #     success rate.
    #     """
    #     attack = ZooContainer(
    #         self.mc,
    #         targeted=False,
    #         learning_rate=1e-2,
    #         max_iter=15,
    #         binary_search_steps=5,
    #         abort_early=True,
    #         use_resize=False,
    #         use_importance=False,
    #     )
    #     # Slow algorithm, only test 3 samples
    #     adv, y_adv, x_clean, y_clean = attack.generate(count=3)
    #     accuracy = self.mc.evaluate(adv, y_clean)
    #     logger.info('Accuracy on adv. examples: %f', accuracy)

    #     # self.assertFalse((adv == x_clean).all())
    #     # NOTE: Success rate closes to 0
    #     # self.assertGreaterEqual(
    #     #     (y_adv != y_clean).sum() / x_clean.shape[0], 0.9)
    #     self.assertLessEqual(np.max(adv), 1.0 + 1e-4)
    #     self.assertGreaterEqual(np.min(adv), 0 - 1e-4)
    #     l2 = np.max(get_l2_norm(adv, x_clean))
    #     logger.info('L2 norm = %f', l2)


if __name__ == '__main__':
    unittest.main()
