import logging
import os
import unittest

import numpy as np

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         DummyAttack, FGSMContainer, SaliencyContainer)
from aad.basemodels import MnistCnnV2, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, get_pt_model_filename, master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
NUM_ADV = 1000  # number of adversarial examples will be generated
NAME = 'MNIST'
SAMPLE_RATIO = 1000 / 6e4
MAX_EPOCHS = 50


class TestApplicabilityDomainMNIST(unittest.TestCase):
    """
    NOTE: 2020/03/29 Switch from using the hidden layer after conv layer to the 
    hidden layer before output layer. 
    """
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=True)

        model = MnistCnnV2()
        logger.info('Using model: %s', model.__class__.__name__)

        cls.mc = ModelContainerPT(model, cls.dc)

        filename = get_pt_model_filename(MnistCnnV2.__name__, NAME, MAX_EPOCHS)
        file_path = os.path.join('save', filename)
        if not os.path.exists(file_path):
            cls.mc.fit(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
            cls.mc.save(filename, overwrite=True)
        else:
            logger.info('Use saved parameters from %s', filename)
            cls.mc.load(file_path)

        accuracy = cls.mc.evaluate(cls.dc.data_test_np, cls.dc.label_test_np)
        logger.info('Accuracy on test set: %f', accuracy)

        hidden_model = model.hidden_model

        logger.info('sample_ratio: %f', SAMPLE_RATIO)
        cls.ad = ApplicabilityDomainContainer(
            cls.mc,
            hidden_model=hidden_model,
            k1=9,
            reliability=1.6,
            sample_ratio=SAMPLE_RATIO,
            confidence=0.9,
            kappa=10,
        )
        cls.ad.fit()

        # shuffle the test set
        x_test = cls.dc.data_test_np
        y_test = cls.dc.label_test_np
        shuffled_indices = np.random.permutation(len(x_test))[:NUM_ADV]
        cls.x = x_test[shuffled_indices]
        cls.y = y_test[shuffled_indices]
        logger.info('# of test set: %d', len(cls.x))

    def setUp(self):
        master_seed(SEED)

    def preform_attack(self, attack, count=NUM_ADV):
        adv, y_adv, x_clean, y_clean = attack.generate(count=count)

        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)

        blocked_indices, x_passed = self.ad.detect(adv, y_adv)
        logger.info('Blocked %d/%d samples from adv. examples',
                    len(blocked_indices), len(adv))

        missed_indices = np.where(y_adv != y_clean)[0]
        intersect = np.intersect1d(blocked_indices, missed_indices)
        logger.info('# of blocked successful adv. examples: %d',
                    len(intersect))

        passed_indices = np.delete(np.arange(len(adv)), blocked_indices)
        passed_y_clean = y_clean[passed_indices]
        accuracy = self.mc.evaluate(x_passed, passed_y_clean)
        logger.info('Accuracy on passed adv. examples: %f', accuracy)
        return blocked_indices

    def test_block_clean(self):
        dummy_attack = DummyAttack(self.mc)
        blocked_indices = self.preform_attack(dummy_attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertLessEqual(block_rate, 0.069)
        logger.info('Block rate: %f', block_rate)

    def test_block_train(self):
        n = NUM_ADV
        x = self.dc.data_train_np
        y = self.dc.label_train_np
        shuffled_indices = np.random.permutation(len(x))[:n]
        x = x[shuffled_indices]
        y = y[shuffled_indices]

        blocked_indices, x_passed = self.ad.detect(x)
        print(f'# of blocked: {len(blocked_indices)}')
        self.assertEqual(len(x_passed) + len(blocked_indices), n)
        block_rate = len(blocked_indices) / n
        self.assertLessEqual(block_rate, 0.045)
        logger.info('Block rate: %f', block_rate)

    def test_fgsm_attack(self):
        """
        NOTE: Fail: 0.786 not greater than or equal to 0.798
        """
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True)
        blocked_indices = self.preform_attack(attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreaterEqual(block_rate, 0.798)
        logger.info('[%s] Block rate: %f', FGSMContainer.__name__, block_rate)

    def test_bim_attack(self):
        attack = BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False)
        blocked_indices = self.preform_attack(attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreaterEqual(block_rate, 0.64)
        logger.info('[%s] Block rate: %f', BIMContainer.__name__, block_rate)

    def test_deepfool_attack(self):
        attack = DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10)
        blocked_indices = self.preform_attack(attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, 0.53)
        logger.info('[%s] Block rate: %f',
            DeepFoolContainer.__name__, block_rate)

    def test_carlini_l2_attack(self):
        n = 100
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
            batch_size=16)
        blocked_indices = self.preform_attack(attack, count=n)
        block_rate = len(blocked_indices) / n
        self.assertGreater(block_rate, 0.7)
        logger.info('[%s] Block rate: %f',
                    CarliniL2Container.__name__, block_rate)

    # def test_zoo_attack(self):
    #     """
    #     The attack algorithm has los success rate
    #     """
    #     n = 100
    #     attack = ZooContainer(
    #         self.mc,
    #         targeted=False,
    #         max_iter=10,
    #         binary_search_steps=3,
    #         abort_early=False,
    #         use_resize=False,
    #         use_importance=False,
    #     )
    #     blocked_indices = self.preform_attack(attack, count=n)
    #     blocked_rate = len(blocked_indices) / n
    #     self.assertGreater(blocked_rate, 0.5)

    def test_saliency_attack(self):
        n = 100
        attack = SaliencyContainer(
            self.mc,
        )
        blocked_indices = self.preform_attack(attack, count=n)
        block_rate = len(blocked_indices) / n
        # tested block rate 80%
        self.assertGreater(block_rate, 0.79)
        logger.info('[%s] Block rate: %f',
            SaliencyContainer.__name__, block_rate)


if __name__ == '__main__':
    unittest.main()
