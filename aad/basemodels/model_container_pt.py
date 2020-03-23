
"""
This module implements the base class for PyTorch model container.
"""
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ..datasets import DataContainer
from ..utils import name_handler, swap_image_channel

logger = logging.getLogger(__name__)


class ModelContainerPT:
    def __init__(self, model, data_container):
        assert isinstance(model, nn.Module), \
            f'Expecting a Torch Module, got {type(model)}'
        self.model = model
        assert isinstance(data_container, DataContainer), \
            f'Expectiong a DataContainer, got {type(data_container)}'
        self.data_container = data_container

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.debug('Using device: %s', self.device)

        # to allow the model train multiple times
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []

    def fit(self, epochs=5, batch_size=64):
        since = time.time()

        self._fit_torch(epochs, batch_size)

        time_elapsed = time.time() - since
        logger.info('Time to complete training: %im %.3fs',
                    int(time_elapsed // 60), time_elapsed % 60)

    def save(self, filename, overwrite=False):
        filename = name_handler(filename, 'pt', overwrite)
        filename = os.path.join('save', filename)

        torch.save(self.model.state_dict(), filename)

        logger.info('Successfully saved model to %s', filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(
            filename, map_location=self.device))

        logger.info('Successfully loaded model from %s', filename)

    def score(self, x):
        if not isinstance(x, torch.Tensor):
            # only swap the channel when it is wrong!
            if self.data_container.data_type == 'image' and x.shape[1] not in (1, 3):
                x = swap_image_channel(x)
            x = torch.from_numpy(x)
        x = x.float().to(self.device)
        return self.model(x).cpu().detach().numpy()

    def predict(self, x, require_score=False):
        if not isinstance(x, torch.Tensor):
            # only swap the channel when it is wrong!
            if self.data_container.data_type == 'image' and x.shape[1] not in (1, 3):
                x = swap_image_channel(x)
            x = torch.from_numpy(x)
        x = x.float().to(self.device)
        outputs = self.model(x)
        predictions = outputs.max(1, keepdim=True)[1]
        predictions = predictions.cpu().detach().numpy().squeeze()

        if require_score:
            outputs = outputs.cpu().detach().numpy()
            return predictions, outputs
        else:
            return predictions

    def predict_one(self, x, require_score=False):
        if isinstance(x, np.ndarray):
            x = np.expand_dims(x, axis=0)
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(dim=0)
        else:
            raise TypeError(f'Got {type(x)}. Except a ndarray or tensor')
        prediction, output = self.predict(x, True)
        if require_score:
            return prediction.squeeze(), output.squeeze()
        else:
            return prediction.squeeze()

    def evaluate(self, x, labels):
        predictions = self.predict(x)
        accuracy = np.sum(np.equal(predictions, labels)) / len(labels)
        return accuracy

    def _fit_torch(self, epochs, batch_size):

        train_loader = self.data_container.get_dataloader(
            batch_size, is_train=True)
        test_loader = self.data_container.get_dataloader(
            batch_size, is_train=False)

        # parameters are passed as dict, so it allows different optimizer
        params = self.model.optim_params
        optimizer = self.model.optimizer(self.model.parameters(), **params)

        # scheduler is optional
        if self.model.scheduler:
            scheduler_params = self.model.scheduler_params
            scheduler = self.model.scheduler(optimizer, **scheduler_params)

        for epoch in range(epochs):
            time_start = time.time()

            tr_loss, tr_acc = self._train_torch(optimizer, train_loader)
            va_loss, va_acc = self._validate_torch(test_loader)
            if self.model.scheduler:
                scheduler.step()

            time_elapsed = time.time() - time_start
            logger.debug(
                '[%i/%i]%im %.3fs: Train loss: %f acc: %f - Test loss: %f acc: %f',
                epoch+1, epochs,
                int(time_elapsed // 60), time_elapsed % 60,
                tr_loss, tr_acc, va_loss, va_acc)

            # save logs
            self.loss_train.append(tr_loss)
            self.loss_test.append(va_loss)
            self.accuracy_train.append(tr_acc)
            self.accuracy_test.append(va_acc)

            # early stopping
            if tr_acc >= 0.999 and va_acc >= 0.999:
                logger.debug(
                    'Satisfied the accuracy threshold. Abort at %i epoch!',
                    epoch)
                break

    def _train_torch(self, optimizer, loader):
        self.model.train()
        total_loss = 0.
        corrects = 0.
        loss_fn = self.model.loss_fn

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = x.size(0)

            optimizer.zero_grad()
            output = self.model(x)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            # for logging
            total_loss += loss.item() * batch_size
            predictions = output.max(1, keepdim=True)[1]
            corrects += predictions.eq(y.view_as(predictions)).sum().item()

        n = len(loader.dataset)
        total_loss = total_loss / n
        acc = corrects / n
        return total_loss, acc

    def _validate_torch(self, loader):
        self.model.eval()
        total_loss = 0.
        corrects = 0
        loss_fn = self.model.loss_fn

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)
                output = self.model(x)
                loss = loss_fn(output, y)
                total_loss += loss.item() * batch_size
                predictions = output.max(1, keepdim=True)[1]
                corrects += predictions.eq(y.view_as(predictions)).sum().item()

        n = len(loader.dataset)
        total_loss = total_loss / n
        accuracy = corrects / n
        return total_loss, accuracy
