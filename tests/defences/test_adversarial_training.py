import logging
import os
import unittest

import numpy as np

from aad.attacks import BIMContainer, DeepFoolContainer, FGSMContainer
from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import AdversarialTraining
from aad.utils import get_data_path, master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
NAME = 'MNIST'
MAX_EPOCHS = 10
RATIO = 0.1
MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')
ADV_FILE = os.path.join('save', 'MnistCnnV2_MNIST_FGSM_adv.npy')
ADV_TRAINER_FILE = os.path.join('test', 'test_advtr_MnistCnnV2_MNIST_e10.pt')


class TestAdversarialTraining(unittest.TestCase):
    """Testing Adversarial Training as Defence."""

    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        Model = get_model('MnistCnnV2')
        model = Model()
        logger.info('Starting %s data container...', NAME)
        dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        dc()
        cls.mc = ModelContainerPT(model, dc)
        cls.mc.load(MODEL_FILE)
        accuracy = cls.mc.evaluate(dc.x_test, dc.y_test)
        logger.info('Accuracy on test set: %f', accuracy)

        cls.attack = BIMContainer(
            cls.mc,
            eps=0.3,
            eps_step=0.1,
            max_iter=100,
            targeted=False)

    def setUp(self):
        master_seed(SEED)

    def test_fit_save(self):
        adv_trainer = AdversarialTraining(self.mc, [self.attack])
        adv_trainer.fit(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            ratio=RATIO)
        adv_trainer.save(ADV_TRAINER_FILE, overwrite=True)

    def test_load(self):
        adv_trainer = AdversarialTraining(self.mc, [self.attack])
        adv_trainer.load(os.path.join('save', ADV_TRAINER_FILE))
        robust_model = adv_trainer.get_def_model_container()
        dc = self.mc.data_container
        x = dc.x_test
        y = dc.y_test
        accuracy = robust_model.evaluate(x, y)
        logger.info('Accuracy on test set: %f', accuracy)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_train_no_attack(self):
        adv_trainer = AdversarialTraining(self.mc)
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            targeted=False,
            batch_size=128)
        dc = self.mc.data_container
        x_train = np.copy(dc.x_train)
        y_train = dc.y_train
        indices = np.random.choice(
            np.arange(len(x_train)),
            int(np.floor(len(x_train) * 0.2)),
            replace=False)
        adv, y_adv, x, y = attack.generate(
            use_testset=False, x=x_train[indices])
        x_train[indices] = adv
        adv_trainer.fit_discriminator(x_train, y_train, MAX_EPOCHS, BATCH_SIZE)
        robust_model = adv_trainer.get_def_model_container()
        accuracy = robust_model.evaluate(adv, y)
        logger.info('Accuracy on FGSM set: %f', accuracy)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_multi_attacks(self):
        attack2 = DeepFoolContainer(
            self.mc,
            max_iter=100,
            epsilon=1e-6,
            nb_grads=10,
            batch_size=16)
        adv_trainer = AdversarialTraining(self.mc, [self.attack, attack2])
        adv_trainer.fit(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            ratio=RATIO)
        attack3 = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            targeted=False,
            batch_size=128)
        adv, y_adv, x, y = attack3.generate(count=1000, use_testset=True)
        robust_model = adv_trainer.get_def_model_container()
        accuracy = robust_model.evaluate(adv, y)
        logger.info('Accuracy on FGSM set: %f', accuracy)
        self.assertGreaterEqual(accuracy, 0.6)

    def test_detect(self):
        adv_trainer = AdversarialTraining(self.mc, [self.attack])
        adv_trainer.load(os.path.join('save', ADV_TRAINER_FILE))
        robust_model = adv_trainer.get_def_model_container()

        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            targeted=False,
            batch_size=128)
        adv, y_adv, x, y = attack.generate(count=1000, use_testset=True)
        blocked_indices, x_passed = adv_trainer.detect(
            adv, y_adv, return_passed_x=True)
        self.assertGreaterEqual(len(blocked_indices), 700)
        logger.info('blocked adversarial: %d', len(blocked_indices))

        # blocked + passed = full set
        self.assertEqual(len(blocked_indices) + len(x_passed), len(adv))

        # accuracy on blind model
        accuracy_before = self.mc.evaluate(adv, y)
        logger.info('Accuracy of adv. examples on blind model: %f',
                    accuracy_before)

        # accuracy on passed adv
        passed_indices = np.where(
            np.isin(np.arange(len(adv)), blocked_indices) == False)[0]
        self.assertEqual(len(passed_indices), len(x_passed))
        y_passed = y[passed_indices]
        accuracy = robust_model.evaluate(x_passed, y_passed)
        logger.info('Accuracy on passed adv. examples: %f', accuracy)
        self.assertGreaterEqual(accuracy, accuracy_before)

        # detect clean set
        blocked_indices, x_passed = adv_trainer.detect(
            x, y, return_passed_x=True)
        logger.info('blocked clean samples: %d', len(blocked_indices))
        self.assertLessEqual(len(blocked_indices), len(adv) * 0.1)

        # comparison between accuracy on robust model and accuracy on blind model
        accuracy_robust = robust_model.evaluate(x, y)
        logger.info('Accuracy of clean samples from robust model: %f', accuracy_robust)
        accuracy_blind = self.mc.evaluate(x, y)
        logger.info('Accuracy of clean samples from blind model: %f', accuracy_blind)
        self.assertGreater(accuracy_robust, accuracy_blind * 0.95)


if __name__ == '__main__':
    unittest.main()
