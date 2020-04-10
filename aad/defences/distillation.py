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

    def __init__(self,
                 model_container,
                 distillation_model,
                 temperature=10.0,
                 pretrained=False):
        """
        Create an instance of Distillation Container.

        :param model_container: A trained model.
        :type model_container: `ModelContainerPT`
        :param distillation_model: The distillation model has same architecture as the classifier model.
        :type distillation_model: `torch.nn.Module`
        :param temperature: It controls the smoothness of the softmax function.
        :type temperature: `float`
        :param pretrained: For testing only. Load the pre-trained parameters. The accuracy on train set will become 100%.
        :type pretrained: `bool`
        """
        super(DistillationContainer, self).__init__(model_container)

        self.temperature = temperature
        self.pretrained = pretrained

        # check if the model produces probability outputs
        dc = self.model_container.data_container
        accuracy = self.model_container.evaluate(
            dc.data_test_np, dc.label_test_np)
        logger.debug('Test set accuracy on pre-trained model: %f', accuracy)
        score_train = self.model_container.get_score(dc.data_train_np)
        are_probability = np.all([is_probability(yy) for yy in score_train])
        # We do NOT need soft label for test set.
        # NOTE: What about missclassification?
        if not are_probability:
            prob_train = torch.softmax(
                torch.from_numpy(score_train) / temperature, dim=1).numpy()
        else:
            prob_train = score_train

        labels = np.argmax(prob_train, axis=1)
        correct = len(np.where(labels == dc.label_train_np)[0])
        logger.debug('Accuracy of smooth labels: %f',
                     correct / len(dc.label_train_np))

        # create new data container and replace the label to smooth probability
        dataset_name = dc.name
        dc = DataContainer(DATASET_LIST[dataset_name], get_data_path())
        dc(shuffle=False)
        # prevent the train set permutate.
        dc.data_train_np = self.model_container.data_container.data_train_np
        dc.label_train_np = prob_train

        # smooth model
        smooth_model = distillation_model

        # load pre-trained parameters
        if pretrained:
            state_dict = copy.deepcopy(self.model_container.model.state_dict())
            smooth_model.load_state_dict(state_dict)

        # model container for distillation
        self.smooth_mc = ModelContainerPT(smooth_model, dc)

        accuracy = self.smooth_mc.evaluate(dc.data_train_np, labels)
        logger.debug('Train set accuracy on distillation model: %f', accuracy)
        accuracy = self.smooth_mc.evaluate(dc.data_test_np, dc.label_test_np)
        logger.debug('Test set accuracy on distillation model: %f', accuracy)

        # to allow the model train multiple times
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []

    def fit(self, max_epochs=10, batch_size=128):
        """Train the distillation model."""
        mc = self.smooth_mc
        dc = mc.data_container
        # Train set: y is soft probabilities
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

    def save(self, filename, overwrite=False):
        """Save trained parameters."""
        self.smooth_mc.save(filename, overwrite)

    def load(self, filename):
        """Load pre-trained parameters."""
        self.smooth_mc.load(filename)

    def detect(self, adv, pred=None):
        """
        Compare the prediction with original model, and block all unmatched results.
        """
        if pred is None:
            pred = self.model_container.predict(adv)

        distill_pred = self.smooth_mc.predict(adv)
        blocked_indices = np.where(distill_pred != pred)[0]
        return blocked_indices

    def get_def_model_container(self):
        """Get the defence model container."""
        return self.smooth_mc

    def _train(self, optimizer, loader):
        mc = self.smooth_mc
        device = mc.device
        model = mc.model
        model.train()
        total_loss = 0.0
        corrects = 0.0
        loss_fn = self.smooth_nlloss

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            optimizer.zero_grad()
            score = model(x)
            loss = loss_fn(score, y)

            loss.backward()
            optimizer.step()

            # for logging
            total_loss += loss.cpu().item() * batch_size
            pred = score.max(1, keepdim=True)[1]

            labels = y.max(1, keepdim=True)[1]
            corrects += pred.eq(labels).sum().cpu().item()

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
                score = model(x)
                loss = loss_fn(score, y)
                total_loss += loss.cpu().item() * batch_size
                pred = score.max(1, keepdim=True)[1]
                corrects += pred.eq(y.view_as(pred)).sum().cpu().item()

        n = len(loader.dataset)
        total_loss = total_loss / n
        accuracy = corrects / n
        return total_loss, accuracy

    @staticmethod
    def smooth_nlloss(score, targets):
        """
        The loss function for distillation. It's a modified negative log
        likelihood loss.
        """
        logsoftmax = nn.LogSoftmax(dim=1)
        score = logsoftmax(score)
        n = score.size(0)
        loss = - (targets * score).sum() / n
        # targets = targets.max(1, keepdim=True)[1]
        # loss = nn.functional.nll_loss(score, targets.squeeze())
        return loss
