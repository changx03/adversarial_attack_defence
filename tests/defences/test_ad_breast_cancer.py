import logging
import os
import unittest

import numpy as np

from aad.attacks import (BIMContainer, CarliniL2Container, DeepFoolContainer,
                         DummyAttack, FGSMContainer)
from aad.basemodels import BCNN, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, get_pt_model_filename, master_seed

logger = logging.getLogger(__name__)

SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 256  # Train the entire set in one batch
NUM_ADV = 100  # number of adversarial examples will be generated
NAME = 'BreastCancerWisconsin'
MAX_EPOCHS = 200
SAMPLE_RATIO = 1.0


class TestApplicabilityDomainIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        logger.info('Starting %s data container...', NAME)
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        # ordered by labels, it requires shuffle!
        cls.dc(shuffle=True, normalize=True)

        num_features = cls.dc.dim_data[0]
        num_classes = cls.dc.num_classes
        model = BCNN(num_features, num_classes)
        logger.info('Using model: %s', model.__class__.__name__)

        cls.mc = ModelContainerPT(model, cls.dc)

        filename = get_pt_model_filename(BCNN.__name__, NAME, MAX_EPOCHS)
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
        cls.ad = ApplicabilityDomainContainer(
            cls.mc,
            hidden_model=hidden_model,
            k1=6,
            reliability=1.6,
            sample_ratio=SAMPLE_RATIO,
            confidence=0.9,
            kappa=10,
        )
        cls.ad.fit()

    def setUp(self):
        master_seed(SEED)

    def preform_attack(self, attack, count=NUM_ADV):
        adv, y_adv, x_clean, y_clean = attack.generate(count=count)
        not_match = y_adv != y_clean
        adv_success_rate = len(not_match[not_match==True]) / len(adv)

        accuracy = self.mc.evaluate(adv, y_clean)
        logger.info('Accuracy on adv. examples: %f', accuracy)

        x_passed, blocked_indices = self.ad.detect(adv, y_adv)
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
        dummy_attack = DummyAttack(self.mc)
        blocked_indices, _ = self.preform_attack(dummy_attack)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertLessEqual(block_rate, 0.15)
        logger.info('Block rate: %f', block_rate)

    def test_block_train(self):
        n = NUM_ADV
        x = self.dc.data_train_np
        y = self.dc.label_train_np
        shuffled_indices = np.random.permutation(len(x))[:n]
        x = x[shuffled_indices]
        y = y[shuffled_indices]

        x_passed, blocked_indices = self.ad.detect(x)
        print(f'# of blocked: {len(blocked_indices)}')
        self.assertEqual(len(x_passed) + len(blocked_indices), n)
        block_rate = len(blocked_indices) / n
        self.assertLessEqual(block_rate, 0.15)
        logger.info('Block rate: %f', block_rate)

    def test_fgsm_attack(self):
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.01,
            minimal=True)
        blocked_indices, adv_success_rate = self.preform_attack(attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.8)
        logger.info('[%s] Block rate: %f', FGSMContainer.__name__, block_rate)

    def test_bim_attack(self):
        """
        NOTE: Unable to pass the test
        """
        attack = BIMContainer(
            self.mc,
            eps=0.3,
            eps_step=0.01,
            max_iter=100,
            targeted=False)
        blocked_indices, adv_success_rate = self.preform_attack(attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.8)
        logger.info('[%s] Block rate: %f', BIMContainer.__name__, block_rate)

    def test_deepfool_attack(self):
        attack = DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10)
        blocked_indices, adv_success_rate = self.preform_attack(attack, count=NUM_ADV)
        block_rate = len(blocked_indices) / NUM_ADV
        self.assertGreater(block_rate, adv_success_rate * 0.8)
        logger.info('[%s] Block rate: %f',
                    DeepFoolContainer.__name__, block_rate)

    def test_carlini_l2_attack(self):
        """
        Testing defence against Carlini & Wagner L2 attack
        NOTE: This attack outputs a lot of debug lines!
        """
        n = NUM_ADV
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
            batch_size=n)
        blocked_indices, adv_success_rate = self.preform_attack(attack, count=n)
        block_rate = len(blocked_indices) / n
        self.assertGreater(block_rate, adv_success_rate * 0.8)
        logger.info('[%s] Block rate: %f',
                    CarliniL2Container.__name__, block_rate)


if __name__ == '__main__':
    unittest.main()
