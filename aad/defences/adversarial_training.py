"""
This module implements adversarial training
"""
import copy
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..attacks import AttackContainer
from ..basemodels import ModelContainerPT
from ..datasets import GenericDataset
from ..utils import swap_image_channel
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class AdversarialTraining(DetectorContainer):
    """
    Class performs Adversarial Training.
    """

    def __init__(self,
                 model_container,
                 attacks=[]):
        """
        Create an AdversarialTraining class instance.

        Parameters
        ----------
        model_container : ModelContainerPT
            Pre-trained classification model.
        attacks : list of AttackContainer
            The adversarial attacks for data augmentation in adversarial training.
        """
        super(AdversarialTraining, self).__init__(model_container)
        self._attacks = attacks

        if not isinstance(model_container, ModelContainerPT):
            raise ValueError(
                'model_container is not an instance of ModelContainerPT.')
        if not np.all([isinstance(att, AttackContainer) for att in attacks]):
            raise ValueError('attacks is not a list of AttackContainer.')

        self._discriminator = copy.deepcopy(model_container)

        # place holder for parameters
        self._params = {
            'max_epochs': 0,
            'batch_size': None,
            'ratio': None,
        }

    @property
    def attacks(self):
        return self._attacks

    def fit(self, max_epochs=10, batch_size=128, ratio=0.2):
        """
        Train the classifier with adversarial examples.

        Parameters
        ----------
        max_epochs : int
            Number of epochs the program will run during the training.
        batch_size : int
            Size of a mini-batch.
        ratio : float
            The percentage of train set will be used for generating adversarial examples.
        """
        if len(self._attacks) == 0:
            logger.warning(
                'No adversarial attack is available. Consider call fit_discriminator instead.')
            return

        params = {'ratio': ratio}
        self.set_params(**params)

        # generate adversarial examples
        dc = self.model_container.data_container
        x_train = np.copy(dc.x_train)
        num_train = len(x_train)
        # create an index pool for adversarial examples
        num_adv = int(np.floor(num_train * ratio))
        adv_indices = np.random.choice(
            list(range(num_train)),
            num_adv,
            replace=False)
        pool_size = num_adv // len(self._attacks)

        self._log_time_start()
        start = 0
        for i in range(len(self._attacks)):
            end = start + pool_size
            if i == len(self._attacks) - 1:
                end = num_adv
            indices = adv_indices[start: end]

            x = x_train[indices]
            attack = self._attacks[i]
            logger.debug('Generate %d adv. examples using [%s]',
                         end-start, attack.__class__.__name__,)
            adv, y_adv, x_clean, y_clean = attack.generate(
                use_testset=False, x=x)
            x_train[indices] = adv
            start += end
        self._log_time_end('Train Adv. Examples')

        # train
        y_train = dc.y_train
        self.fit_discriminator(x_train, y_train, max_epochs, batch_size)

    def save(self, filename, overwrite=False):
        """Save trained parameters."""
        self._discriminator.save(filename, overwrite)

    def load(self, filename):
        """Load pre-trained parameters."""
        self._discriminator.load(filename)

    def detect(self, adv, pred=None, return_passed_x=False):
        """
        Compare the predictions between adv. training model and original model and block all unmatched results.

        Parameters
        ----------
        adv : numpy.ndarray
            The data for evaluation.
        pred : numpy.ndarray, optional
            The predictions of the input data. If it is none, this method will use internal model to make prediction.
        return_passed_x : bool
            The flag of returning the data which are passed the test.

        Returns
        -------
        block_indices : numpy.ndarray
            List of blocked indices.
        passed_x : numpy.ndarray
            The data which are passed the test. This parameter will not be returns if `return_passed_x` is False.
        """
        if pred is None:
            pred = self.model_container.predict(adv)

        robust_pred = self._discriminator.predict(adv)
        blocked_indices = np.where(robust_pred != pred)[0]

        if return_passed_x:
            passed_indices = np.where(
                np.isin(np.arange(len(adv)), blocked_indices) == False)[0]
            return blocked_indices, adv[passed_indices]
        return blocked_indices

    def fit_discriminator(self, x_train, y_train, max_epochs, batch_size):
        """
        Train the model with an extra train set.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        max_epochs : int
            Number of epochs the program will run during the training.
        batch_size : int
            Size of a mini-batch.
        """
        mc = self._discriminator
        dc = mc.data_container
        test_loader = dc.get_dataloader(batch_size, is_train=False)
        model = mc.model

        # Build a train dataset with adversarial examples.
        if dc.data_type == 'image' and x_train.shape[1] not in (1, 3):
            x_train = swap_image_channel(x_train)

        dataset = GenericDataset(x_train, y_train)
        train_loader = DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=0)

        best_model_state = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # parameters are passed as dict, so it allows different optimizer
        params = model.optim_params
        optimizer = model.optimizer(model.parameters(), **params)

        # scheduler is optional
        if model.scheduler:
            scheduler_params = model.scheduler_params
            scheduler = model.scheduler(optimizer, **scheduler_params)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for epoch in range(max_epochs):
            time_start = time.time()

            tr_loss, tr_acc = mc._train_torch(optimizer, train_loader)
            va_loss, va_acc = mc._validate_torch(test_loader)
            if model.scheduler:
                scheduler.step()

            time_elapsed = time.time() - time_start
            logger.debug(
                '[%d/%d]%dm %.3fs: Train loss: %f acc: %f - Test loss: %f acc: %f',
                epoch+1, max_epochs,
                int(time_elapsed // 60), time_elapsed % 60,
                tr_loss, tr_acc, va_loss, va_acc)

            # save best state
            if va_acc >= best_acc:
                best_acc = va_acc
                best_model_state = copy.deepcopy(model.state_dict())

            # save logs
            mc.loss_train.append(tr_loss)
            mc.loss_test.append(va_loss)
            mc.accuracy_train.append(tr_acc)
            mc.accuracy_test.append(va_acc)

            # early stopping
            if (tr_acc >= 0.999 and va_acc >= 0.999) or tr_loss < 1e-4:
                logger.debug(
                    'Satisfied the accuracy threshold. Abort at %d epoch!',
                    epoch)
                break
            if len(mc.loss_train) - 5 >= 0 \
                    and mc.loss_train[-5] <= mc.loss_train[-1]:
                logger.debug(
                    'No improvement in the last 5 epochs. Abort at %d epoch!',
                    epoch)
                break

        model.load_state_dict(best_model_state)

    def get_def_model_container(self):
        """
        Get the discriminator model container.

        Returns
        -------
        ModelContainer
            The discriminator model container.
        """
        return self._discriminator
