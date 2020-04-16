import logging
import os
import unittest

import numpy as np

import aad.attacks as attacks
from aad.basemodels import IrisNN, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import (get_data_path, get_pt_model_filename, get_range,
                       master_seed)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 256  # Train the entire set in one batch
NUM_ADV = 50  # number of adversarial examples will be generated
NAME = 'Iris'
MAX_EPOCHS = 300
SAMPLE_RATIO = 1.0


class TestApplicabilityDomainIris(unittest.TestCase):
    """Testing Applicability Domain on Iris dataset"""

    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        # ordered by labels, it requires shuffle!
        cls.dc(shuffle=True, normalize=True, size_train=0.6)

        model = IrisNN(hidden_nodes=12)
        logger.info('Using model: %s', model.__class__.__name__)

        cls.mc = ModelContainerPT(model, cls.dc)

        filename = get_pt_model_filename(IrisNN.__name__, NAME, MAX_EPOCHS)
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
        cls.ad = ApplicabilityDomainContainer(
            cls.mc,
            hidden_model=hidden_model,
            k1=6,
            reliability=1.6,
            sample_ratio=SAMPLE_RATIO,
            confidence=0.76,
            kappa=10,
        )
        cls.ad.fit()

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
        blocked_indices, _ = self.preform_attack(dummy_attack)
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
            eps_step=0.01,
            minimal=True)
        blocked_indices, adv_success_rate = self.preform_attack(
            attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.6)
        logger.info(
            '[%s] Block rate: %f',
            attack.__class__.__name__,
            block_rate)

    def test_bim_attack(self):
        attack = attacks.BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.01,
            max_iter=1000,
            targeted=False)
        blocked_indices, adv_success_rate = self.preform_attack(
            attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.6)
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
        blocked_indices, adv_success_rate = self.preform_attack(
            attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.6)
        logger.info(
            '[%s] Block rate: %f',
            attack.__class__.__name__,
            block_rate)

    def test_carlini_l2_attack(self):
        clip_values = get_range(self.dc.x_train)
        # Lower the upper bound of `c_range` will reduce the norm of perturbation.
        attack = attacks.CarliniL2V2Container(
            self.mc,
            learning_rate=0.01,
            binary_search_steps=9,
            max_iter=1000,
            confidence=0.0,
            initial_const=0.01,
            c_range=(0, 1e4),
            batch_size=16,
            clip_values=clip_values
        )
        blocked_indices, adv_success_rate = self.preform_attack(
            attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.6)
        logger.info('[%s] Block rate: %f',
                    attack.__class__.__name__, block_rate)


if __name__ == '__main__':
    unittest.main()
