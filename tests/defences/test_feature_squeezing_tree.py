import logging
import os
import unittest

import numpy as np
from art.attacks import DecisionTreeAttack
from art.classifiers import SklearnClassifier
from sklearn.datasets import make_classification
from sklearn.tree import ExtraTreeClassifier

from aad.attacks import FGSMContainer
from aad.basemodels import ModelContainerTree
from aad.datasets import DataContainer, get_synthetic_dataset_dict
from aad.defences import FeatureSqueezingTree
from aad.utils import (get_data_path, get_l2_norm, get_range, master_seed,
                       scale_normalize)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
SAMPLE_SIZE = 5000
NUM_FEATURES = 8
NUM_CLASSES = 2
MODEL_FILE = os.path.join('save', 'IrisNN_Iris_e200.pt')
SQUEEZER_FILE = os.path.join('test', 'IrisNN_Iris_e200')


class TestFeatureSqueezing(unittest.TestCase):
    """Testing Feature Squeezing as Defence on Iris dataset."""

    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        # generating synthetic data
        x, y = make_classification(
            n_samples=SAMPLE_SIZE,
            n_features=NUM_FEATURES,
            n_informative=NUM_CLASSES,
            n_redundant=0,
            n_classes=NUM_CLASSES,
            n_clusters_per_class=1,
        )
        x_max = np.max(x, axis=0)
        x_min = np.min(x, axis=0)
        x = scale_normalize(x, x_min, x_max)
        n_train = int(np.floor(SAMPLE_SIZE * 0.8))
        x_train = np.array(x[:n_train], dtype=np.float32)
        y_train = np.array(y[:n_train], dtype=np.long)
        x_test = np.array(x[n_train:], dtype=np.float32)
        y_test = np.array(y[n_train:], dtype=np.long)

        data_dict = get_synthetic_dataset_dict(
            SAMPLE_SIZE, NUM_CLASSES, NUM_FEATURES)
        dc = DataContainer(data_dict, get_data_path())
        dc.x_train = x_train
        dc.y_train = y_train
        dc.x_test = x_test
        dc.y_test = y_test

        # training Extra Tree classifier
        classifier = ExtraTreeClassifier(
            criterion='gini',
            splitter='random',
        )
        cls.mc = ModelContainerTree(classifier, dc)
        cls.mc.fit()
        accuracy = cls.mc.evaluate(dc.x_test, dc.y_test)
        logger.info('Accuracy on test set: %f', accuracy)

    def setUp(self):
        master_seed(SEED)

    def test_data(self):
        x_train = np.copy(self.mc.data_container.x_train)
        y_train = np.copy(self.mc.data_container.y_train)
        squeezer = FeatureSqueezingTree(
            self.mc,
            smoothing_methods=['normal', 'binary'],
            bit_depth=8,
            sigma=0.1,
            kernel_size=2,
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

        squeezer.fit()

        # expecting test set and train set have not been altered.
        res = np.all(squeezer.model_container.data_container.x_test == x_test)
        self.assertTrue(res)
        res = np.all(squeezer.model_container.data_container.y_test == y_test)
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

        squeezer = FeatureSqueezingTree(
            self.mc,
            clip_values=x_range,
            smoothing_methods=['normal', 'binary'],
            bit_depth=8,
            sigma=0.1,
            kernel_size=2,
            pretrained=False,
        )

        # Expecting difference between input data and squeezed data
        mc_normal = squeezer.get_def_model_container('normal')
        mc_binary = squeezer.get_def_model_container('binary')

        self.assertFalse((x_train == mc_normal.data_container.x_train).all())
        self.assertFalse((x_train == mc_binary.data_container.x_train).all())

        # average perturbation
        l2 = np.mean(get_l2_norm(x_train, mc_normal.data_container.x_train))
        logger.info('L2 norm of normal squeezer:%f', l2)
        self.assertLessEqual(l2, 0.4)

        # maximum perturbation
        l2 = np.max(get_l2_norm(x_train, mc_binary.data_container.x_train))
        logger.info('L2 norm of binary squeezer:%f', l2)
        self.assertLessEqual(l2, 0.4)

    def test_detect(self):
        dc = self.mc.data_container
        # Do not load pretrained parameters
        squeezer = FeatureSqueezingTree(
            self.mc,
            smoothing_methods=['normal', 'binary'],
            bit_depth=8,
            sigma=0.1,
            kernel_size=2,
            pretrained=False,
        )
        squeezer.fit()

        # testing clean set
        x_test = dc.x_test
        pred = self.mc.predict(x_test)
        blocked_indices, passed_x = squeezer.detect(
            x_test, pred,
            return_passed_x=True)
        logger.info('blocked clean samples: %d', len(blocked_indices))
        self.assertLessEqual(len(blocked_indices), 400)
        self.assertEqual(len(blocked_indices) + len(passed_x), len(x_test))

        # the prediction parameter should not alter the result.
        blocked_indices_2, passed_x = squeezer.detect(
            x_test,
            return_passed_x=True)
        self.assertEqual(len(blocked_indices_2) + len(passed_x), len(x_test))

        # test detector using IBM ART's Decision Tree Attack
        art_classifier = SklearnClassifier(self.mc.model)
        attack = DecisionTreeAttack(art_classifier)
        adv = attack.generate(x_test)
        pred_adv = self.mc.predict(adv)
        blocked_indices, adv_passed = squeezer.detect(
            adv, pred_adv, return_passed_x=True)
        logger.info('blocked adversarial: %d', len(blocked_indices))

        passed_indices = np.where(
            np.isin(np.arange(len(adv)), blocked_indices) == False)[0]
        acc_adv = self.mc.evaluate(adv_passed, dc.y_test[passed_indices])
        logger.info('Accuracy on passed adv. examples: %f', acc_adv)
