"""
This module implements the Defensive Distillation method.
"""
import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from ..basemodels import ModelContainerPT
from ..datasets import DATASET_LIST, DataContainer
from ..utils import get_data_path, is_probability
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class DistillationContainer(DetectorContainer):
    """
    Implements the Defensive Distillation method.
    """

    def __init__(self, model_container, temperature=10.0):
        """
        Create an instance of Distillation Container.

        :param model_container: A trained model.
        :type model_container: `ModelContainerPT`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        """
        super(DistillationContainer, self).__init__(model_container)

        self.temperature = temperature

        # check if the model produces probability outputs
        dc = self.model_container.data_container
        score_train = self.model_container.get_score(dc.data_train_np)
        are_probability = np.all([is_probability(yy) for yy in score_train])
        # We do NOT need soft label for test set.
        # NOTE: What about missclassification?
        if not are_probability:
            prob_train = torch.softmax(
                torch.from_numpy(score_train) / temperature, dim=1).numpy()
        else:
            prob_train = score_train

        # smooth model
        Model = self.model_container.model.__class__
        smooth_model = Model(from_logits=True)

        # load pre-trained parameters
        state_dict = copy.deepcopy(self.model_container.model.state_dict())
        smooth_model.load_state_dict(state_dict)

        # create new data container and replace the label to smooth probability
        dataset_name = dc.name
        dc = DataContainer(DATASET_LIST[dataset_name], get_data_path())
        dc()
        dc.label_train_np = prob_train
        self.smooth_mc = ModelContainerPT(smooth_model, dc)

        # to allow the model train multiple times
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []

    def fit(self, max_epochs=10, batch_size=128):
        mc = self.smooth_mc
        dc = mc.data_container
        # Train set: y is soft-labels
        train_loader = dc.get_dataloader(batch_size, is_train=True)
        # Test set: y is hard-labels
        test_loader = dc.get_dataloader(batch_size, is_train=False)
        model = mc.model

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

            tr_loss, tr_acc = self._train(optimizer, train_loader)
            va_loss, va_acc = self._validate(test_loader)
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
            self.loss_train.append(tr_loss)
            self.loss_test.append(va_loss)
            self.accuracy_train.append(tr_acc)
            self.accuracy_test.append(va_acc)

            # early stopping
            if (tr_acc >= 0.999 and va_acc >= 0.999) or tr_loss < 1e-4:
                logger.debug(
                    'Satisfied the accuracy threshold. Abort at %d epoch!',
                    epoch)
                break
            if len(self.loss_train) - 5 >= 0 \
                    and self.loss_train[-5] <= self.loss_train[-1]:
                logger.debug(
                    'No improvement in the last 5 epochs. Abort at %d epoch!',
                    epoch)
                break

        model.load_state_dict(best_model_state)

    def detect(self, adv, pred=None):
        pass

    def get_def_model_container(self):
        """Get defence model container"""
        return self.smooth_mc

    def _train(self, optimizer, loader):
        mc = self.smooth_mc
        device = mc.device
        model = mc.model
        model.train()
        total_loss = 0.0
        corrects = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            optimizer.zero_grad()
            output = model(x)
            loss = self.smooth_nlloss(output, y)

            loss.backward()
            optimizer.step()

            # for logging
            total_loss += loss.cpu().item() * batch_size
            pred = output.max(1, keepdim=True)[1]

            y = y.max(1, keepdim=True)[1]
            corrects += pred.eq(y.view_as(pred)).sum().cpu().item()

        n = len(loader.dataset)
        total_loss = total_loss / n
        acc = corrects / n
        return total_loss, acc

    def _validate(self, loader):
        mc = self.smooth_mc
        device = mc.device
        model = mc.model
        model.eval()
        total_loss = 0.0
        corrects = 0.0
        loss_fn = model.loss_fn

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                batch_size = x.size(0)
                output = model(x)
                loss = loss_fn(output, y)
                total_loss += loss.cpu().item() * batch_size
                pred = output.max(1, keepdim=True)[1]
                corrects += pred.eq(y.view_as(pred)).sum().cpu().item()

        n = len(loader.dataset)
        total_loss = total_loss / n
        accuracy = corrects / n
        return total_loss, accuracy

    @staticmethod
    def smooth_nlloss(inputs, targets):
        n = inputs.size(0)
        logsoftmax = nn.LogSoftmax(dim=1)
        outputs = logsoftmax(inputs)
        loss = - (targets * outputs).sum() / n
        return loss
