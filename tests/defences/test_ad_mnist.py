import logging
import os
import unittest

import numpy as np

import aad.attacks as attacks
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

        accuracy = cls.mc.evaluate(cls.dc.x_test, cls.dc.y_test)
        logger.info('Accuracy on test set: %f', accuracy)

        hidden_model = model.hidden_model

        logger.info('sample_ratio: %f', SAMPLE_RATIO)
        cls.ad = ApplicabilityDomainContainer(
            cls.mc,
            hidden_model=hidden_model,
            k2=9,
            reliability=1.6,
            sample_ratio=SAMPLE_RATIO,
            kappa=10,
            confidence=0.9,
        )
        cls.ad.fit()

        # shuffle the test set
        x_test = cls.dc.x_test
        y_test = cls.dc.y_test
        shuffled_indices = np.random.permutation(len(x_test))[:NUM_ADV]
        cls.x = x_test[shuffled_indices]
        cls.y = y_test[shuffled_indices]
        logger.info('# of test set: %d', len(cls.x))

    def setUp(self):
        master_seed(SEED)

    def preform_attack(self, attack, count=NUM_ADV):
        adv, y_adv, x_clean, y_clean = attack.generate(count=count)
        not_match = y_adv != y_clean
        adv_success_rate = len(not_match[not_match == True]) / len(adv)

        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)

        blocked_indices, x_passed = self.ad.detect(
            adv, y_adv, return_passed_x=True)
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
        return blocked_indices, adv_success_rate

    def test_block_clean(self):
        dummy_attack = attacks.DummyAttack(self.mc)
        blocked_indices, adv_success_rate = self.preform_attack(dummy_attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertLessEqual(block_rate, 0.1)
        logger.info('Block rate: %f', block_rate)

    def test_block_train(self):
        n = NUM_ADV
        x = self.dc.x_train
        y = self.dc.y_train
        shuffled_indices = np.random.permutation(len(x))[:n]
        x = x[shuffled_indices]
        y = y[shuffled_indices]

        blocked_indices, x_passed = self.ad.detect(x, return_passed_x=True)
        print(f'# of blocked: {len(blocked_indices)}')
        self.assertEqual(len(x_passed) + len(blocked_indices), n)
        block_rate = len(blocked_indices) / n
        self.assertLessEqual(block_rate, 0.05)
        logger.info('Block rate: %f', block_rate)

    def test_fgsm_attack(self):
        attack = attacks.FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            minimal=True)
        blocked_indices, adv_success_rate = self.preform_attack(attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreaterEqual(block_rate, adv_success_rate * 0.6)
        logger.info(
            '[%s] Block rate: %f',
            attack.__class__.__name__,
            block_rate)

    def test_bim_attack(self):
        attack = attacks.BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False)
        blocked_indices, adv_success_rate = self.preform_attack(attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreaterEqual(block_rate, adv_success_rate * 0.6)
        logger.info(
            '[%s] Block rate: %f',
            attack.__class__.__name__,
            block_rate)

    def test_deepfool_attack(self):
        attack = attacks.DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10)
        blocked_indices, adv_success_rate = self.preform_attack(attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.6)
        logger.info(
            '[%s] Block rate: %f',
            attack.__class__.__name__,
            block_rate)

    def test_carlini_l2_attack(self):
        attack = attacks.CarliniL2V2Container(
            self.mc,
            learning_rate=0.01,
            binary_search_steps=9,
            max_iter=1000,
            confidence=0.0,
            initial_const=0.01,
            c_range=(0, 1e10),
            batch_size=64,
            clip_values=(0.0, 1.0)
        )
        blocked_indices, adv_success_rate = self.preform_attack(
            attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.6)
        logger.info(
            '[%s] Block rate: %f',
            attack.__class__.__name__,
            block_rate)

    def test_zoo_attack(self):
        pass
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
        attack = attacks.SaliencyContainer(
            self.mc,
        )
        blocked_indices, adv_success_rate = self.preform_attack(
            attack, count=n)
        block_rate = len(blocked_indices) / n
        # tested block rate 80%
        self.assertGreater(block_rate, adv_success_rate * 0.6)
        logger.info('[%s] Block rate: %f',
                    attacks.SaliencyContainer.__name__, block_rate)


if __name__ == '__main__':
    unittest.main()
