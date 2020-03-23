"""
This module implements the base class for adversarial attack.
"""
import abc
import logging
import os

import numpy as np

from ..basemodels import ModelContainerPT
from ..utils import name_handler, onehot_encoding

logger = logging.getLogger(__name__)


class AttackContainer(abc.ABC):
    attack_params = dict()

    def __init__(self, model_containter):
        assert isinstance(model_containter, ModelContainerPT)
        self.model_container = model_containter

    @abc.abstractmethod
    def generate(self, count, use_testset=True, x=None, y=None, **kwargs):
        """Generating adversarial examples."""
        raise NotImplementedError

    def set_params(self, **kwargs):
        """Sets parameters for the attack algorithm."""
        for key, value in kwargs.items():
            if key in self.attack_params.keys():
                self.attack_params[key] = value
        return True

    def predict(self, adv, x):
        """Returns the predictions for adversarial examples and clean inputs."""
        y_adv = self.model_container.predict(adv)
        y = self.model_container.predict(x)
        return y_adv, y

    @staticmethod
    def save_attack(filename, x_adv, y_adv=None, x_clean=None, y_clean=None):
        """Saving adversarial examples."""
        assert isinstance(x_adv, np.ndarray)

        filename_adv = name_handler(filename + '.adv', extension='npy')
        filename_adv = os.path.join('save', filename_adv)

        x_adv.astype(np.float32).tofile(filename_adv)
        if y_adv is not None:
            assert isinstance(y_adv, np.ndarray) \
                and len(y_adv) == len(x_adv)
            filename_y_adv = filename_adv.replace('.adv', '.y_adv')
            y_adv.astype(np.int64).tofile(filename_y_adv)
        if x_clean is not None:
            assert isinstance(x_clean, np.ndarray) \
                and x_adv.shape == x_clean.shape
            filename_raw = filename_adv.replace('.adv', '.x_raw')
            x_clean.astype(np.float32).tofile(filename_raw)
        if y_clean is not None:
            assert isinstance(y_clean, np.ndarray) \
                and len(y_clean) == len(x_adv)
            filename_y = filename_adv.replace('.adv', '.y')
            y_clean.astype(np.int64).tofile(filename_y)
        print(f'Successfully saved model to "{filename_adv}"')

    @staticmethod
    def randam_targets(count, num_classes, use_onehot=False, dtype=np.long):
        """Returns randomly generated labels."""
        y_rand = np.random.choice(num_classes, count, replace=True)
        if not use_onehot:
            return y_rand
        else:
            return onehot_encoding(y_rand, num_classes, dtype)
