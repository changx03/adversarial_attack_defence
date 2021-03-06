import logging
import os
import unittest

import numpy as np

from aad.attacks import FGSMContainer
from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import FeatureSqueezing
from aad.utils import get_data_path, get_l2_norm, get_range, master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
NAME = 'MNIST'
MAX_EPOCHS = 30
MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')
SQUEEZER_FILE = os.path.join('test', 'MnistCnnV2_MNIST_e50')
SQUEEZING_METHODS = ['median']


class TestFeatureSqueezing(unittest.TestCase):
    """Testing Feature Squeezing as Defence on MINIST dataset."""

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

    def setUp(self):
        master_seed(SEED)

    def test_data(self):
        x_train = np.copy(self.mc.data_container.x_train)
        y_train = np.copy(self.mc.data_container.y_train)
        x_range = get_range(self.mc.data_container.x_train)
        squeezer = FeatureSqueezing(
            self.mc,
            clip_values=x_range,
            smoothing_methods=SQUEEZING_METHODS,
            bit_depth=8,
            sigma=0.1,
            kernel_size=3,
            pretrained=True,
        )

        # expecting test set and train set have not been altered.
        res = np.all(
            squeezer.model_container.data_container.x_train == x_train)
        self.assertTrue(res)
        res = np.all(
            squeezer.model_container.data_container.y_train == y_train)
        self.assertTrue(res)

        x_test = np.copy(squeezer.model_container.data_container.x_test)
        y_test = np.copy(squeezer.model_container.data_container.y_test)
        x_train = np.copy(squeezer.model_container.data_container.x_train)
        y_train = np.copy(squeezer.model_container.data_container.y_train)

        squeezer.fit(max_epochs=2, batch_size=128)

        # expecting test set and train set have not been altered.
        res = np.all(
            squeezer.model_container.data_container.x_test == x_test)
        self.assertTrue(res)
        res = np.all(
            squeezer.model_container.data_container.y_test == y_test)
        self.assertTrue(res)
        res = np.all(
            squeezer.model_container.data_container.x_train == x_train)
        self.assertTrue(res)
        res = np.all(
            squeezer.model_container.data_container.y_train == y_train)
        self.assertTrue(res)

    def test_squeezing_data(self):
        x_range = get_range(self.mc.data_container.x_train)
        x_train = np.copy(self.mc.data_container.x_train)

        squeezer = FeatureSqueezing(
            self.mc,
            clip_values=x_range,
            smoothing_methods=SQUEEZING_METHODS,
            bit_depth=8,
            sigma=0.1,
            kernel_size=3,
            pretrained=False,
        )

        # Expecting difference between input data and squeezed data
        mc_median = squeezer.get_def_model_container(SQUEEZING_METHODS[0])

        self.assertFalse((x_train == mc_median.data_container.x_train).all())

        # average perturbation
        # No upper bound on median filter
        l2 = np.mean(get_l2_norm(x_train, mc_median.data_container.x_train))
        logger.info('L2 norm of median squeezer:%f', l2)
        self.assertLessEqual(l2, 10)

    def test_fit_save(self):
        x_range = get_range(self.mc.data_container.x_train)
        squeezer = FeatureSqueezing(
            self.mc,
            clip_values=x_range,
            smoothing_methods=SQUEEZING_METHODS,
            bit_depth=8,
            sigma=0.1,
            kernel_size=3,
            pretrained=False,
        )

        squeezer.fit(max_epochs=MAX_EPOCHS, batch_size=128)

        # save parameters and check the existence of the files
        squeezer.save(SQUEEZER_FILE, True)
        self.assertTrue(os.path.exists(os.path.join(
            'save', 'test', 'MnistCnnV2_MNIST_e50_median.pt')))

    def test_load(self):
        x_range = get_range(self.mc.data_container.x_train)
        # Do not load pretrained parameters
        squeezer = FeatureSqueezing(
            self.mc,
            clip_values=x_range,
            smoothing_methods=SQUEEZING_METHODS,
            bit_depth=8,
            sigma=0.1,
            kernel_size=3,
            pretrained=False,
        )
        squeezer.load(os.path.join('save', SQUEEZER_FILE))

        mc_median = squeezer.get_def_model_container(SQUEEZING_METHODS[0])

        x_test = squeezer.model_container.data_container.x_test
        y_test = squeezer.model_container.data_container.y_test

        acc_med_after = mc_median.evaluate(x_test, y_test)
        logger.info(
            'For median squeezer, accuracy after load parameters: %f',
            acc_med_after)
        self.assertGreater(acc_med_after, 0.90)

    def test_detect(self):
        dc = self.mc.data_container
        x_range = get_range(dc.x_train)
        # Do not load pretrained parameters
        squeezer = FeatureSqueezing(
            self.mc,
            clip_values=x_range,
            smoothing_methods=SQUEEZING_METHODS,
            bit_depth=8,
            sigma=0.1,
            kernel_size=3,
            pretrained=False,
        )
        squeezer.load(os.path.join('save', SQUEEZER_FILE))

        # testing clean set
        x_test = dc.x_test
        pred = self.mc.predict(x_test)
        blocked_indices, passed_x = squeezer.detect(
            x_test, pred, return_passed_x=True)
        logger.info('blocked clean samples: %d', len(blocked_indices))
        self.assertLessEqual(len(blocked_indices), len(x_test) * 0.1)
        self.assertEqual(len(blocked_indices) + len(passed_x), len(x_test))

        # the prediction parameter should not alter the result.
        blocked_indices_2 = squeezer.detect(x_test, return_passed_x=False)
        self.assertTrue((blocked_indices_2 == blocked_indices).all())

        # test detector using FGSM attack
        # NOTE: the block rate is almost 0.
        attack = FGSMContainer(
            self.mc,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            targeted=False,
            batch_size=128)
        adv, y_adv, x, y = attack.generate(count=1000, use_testset=True)
        blocked_indices = squeezer.detect(adv, y_adv, return_passed_x=False)
        logger.info('blocked adversarial: %d', len(blocked_indices))
        self.assertGreater(len(blocked_indices), 500)
