"""
This module implements the base class for adversarial attack.
"""
import abc
import logging
import os
import time

import numpy as np

from ..basemodels import ModelContainerPT
from ..utils import name_handler

logger = logging.getLogger(__name__)


class AttackContainer(abc.ABC):
    _params = dict()  # Override this in child class

    def __init__(self, model_containter):
        assert isinstance(model_containter, ModelContainerPT)
        self.model_container = model_containter
        self._since = 0.0

    @abc.abstractmethod
    def generate(self, count, use_testset=True, x=None, y=None, **kwargs):
        """Generating adversarial examples."""
        raise NotImplementedError

    def set_params(self, **kwargs):
        """Sets parameters for the attack algorithm."""
        for key, value in kwargs.items():
            if key in self._params.keys():
                self._params[key] = value
        return True

    @property
    def attack_params(self):
        return self._params

    def predict(self, adv, x):
        """Returns the predictions for adversarial examples and clean inputs."""
        pred_adv = self.model_container.predict(adv)
        pred_clean = self.model_container.predict(x)
        return pred_adv, pred_clean

    @staticmethod
    def save_attack(filename,
                    x_adv,
                    y_adv=None,
                    x_clean=None,
                    y_clean=None,
                    overwrite=False):
        """Saving adversarial examples."""
        assert isinstance(x_adv, np.ndarray)

        filename = os.path.join('save', filename)
        filename = name_handler(filename + '_[ph]', 'npy', overwrite)
        filename_adv = filename.replace('[ph]', 'adv')

        # x_adv.astype(np.float32).tofile(filename_adv)
        np.save(filename_adv, x_adv.astype(np.float32), allow_pickle=False)
        if y_adv is not None:
            assert isinstance(y_adv, np.ndarray) \
                and len(y_adv) == len(x_adv)
            filename_y_adv = filename.replace('[ph]', 'pred')
            # y_adv.astype(np.int64).tofile(filename_y_adv)
            np.save(filename_y_adv, y_adv.astype(np.int64), allow_pickle=False)
        if x_clean is not None:
            assert isinstance(x_clean, np.ndarray) \
                and x_adv.shape == x_clean.shape
            filename_raw = filename.replace('[ph]', 'x')
            # x_clean.astype(np.float32).tofile(filename_raw)
            np.save(filename_raw, x_clean.astype(
                np.float32), allow_pickle=False)
        if y_clean is not None:
            assert isinstance(y_clean, np.ndarray) \
                and len(y_clean) == len(x_adv)
            filename_y = filename.replace('[ph]', 'y')
            # y_clean.astype(np.int64).tofile(filename_y)
            np.save(filename_y, y_clean.astype(np.int64), allow_pickle=False)
        logger.info('Saved results to %s', filename_adv)

    @staticmethod
    def load_adv_examples(filename):
        """
        Load adversarial examples from a numpy binary file

        Parameters
        ----------
        filename : str
            File name

        Returns
        -------
        adv : numpy.ndarray
            List of adversarial examples.
        pred_adv : numpy.ndarray, optional
            List of predictions of the adversarial examples.
        x_clean : numpy.ndarray, optional
            List of clean inputs for generating the adversarial examples.
        y_true : numpy.ndarray, optional
            List of true labels.
        """
        postfix = ['adv', 'pred', 'x', 'y']
        data_files = [filename.replace('_adv', '_' + s) for s in postfix]
        if not os.path.exists(data_files[1]):
            raise FileExistsError('{} does not exist!'.format(filename))
        adv = np.load(data_files[0], allow_pickle=False)
        pred_adv, x_clean, y_true = None, None, None
        if os.path.exists(data_files[1]):
            pred_adv = np.load(data_files[1], allow_pickle=False)
            assert len(pred_adv) == len(adv)
        if os.path.exists(data_files[2]):
            x_clean = np.load(data_files[2], allow_pickle=False)
            assert x_clean.shape == adv.shape
        if os.path.exists(data_files[3]):
            y_true = np.load(data_files[3], allow_pickle=False)
            assert len(y_true) == len(adv)
        return adv, pred_adv, x_clean, y_true

    def _log_time_start(self):
        self._since = time.time()

    def _log_time_end(self, title=None):
        time_elapsed = time.time() - self._since
        title = ' [' + title + ']' if title else ''
        logger.debug(
            'Time to complete%s: %dm %.3fs',
            title, int(time_elapsed // 60), time_elapsed % 60)
