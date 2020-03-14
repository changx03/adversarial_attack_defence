import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn

from datasets import DataContainer


class TorchModelContainer:
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
        print(f'Using device: {self.device}')

    def fit(self, epochs=5, batch_size=64):
        since = time.time()

        self._fit_torch(epochs, batch_size)

        time_elapsed = time.time() - since
        print('Time taken for training: {:2.0f}m {:2.1f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def save(self, filename, overwrite=False):
        filename = self._name_handler(filename, overwrite)
        torch.save(self.model.state_dict(), filename)

        print(f'Successfully saved model to "{filename}"')

    def load(self, filename):
        self.model.load_state_dict(torch.load(
            filename, map_location=self.device))

        print(f'Successfully loaded model from "{filename}"')

    def pred(self, x, require_output=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        outputs = self.model(x)
        predictions = outputs.max(1, keepdim=True)[1]
        predictions = predictions.cpu().detach().numpy().squeeze()

        if require_output:
            outputs = outputs.cpu().detach().numpy()
            return predictions, outputs
        else:
            return predictions

    def pred_one(self, x, require_output=False):
        prediction, output = self.pred(x.unsqueeze(0), True)
        if require_output:
            return prediction.squeeze(), output.squeeze()
        else:
            return prediction.squeeze()

    def _fit_torch(self, epochs, batch_size):
        # Not all models run multiple iterations
        self.loss_train = np.zeros(epochs, dtype=np.float32)
        self.loss_test = np.zeros(epochs, dtype=np.float32)
        self.accuracy_train = np.zeros(epochs, dtype=np.float32)
        self.accuracy_test = np.zeros(epochs, dtype=np.float32)

        train_loader = self.data_container.get_dataloader(
            batch_size, is_train=True)
        test_loader = self.data_container.get_dataloader(
            batch_size, is_train=False)

        # parameters are passed as dict, so it allows different optimizer
        params = self.model.optim_params
        optimizer = self.model.optimizer(self.model.parameters(), **params)
        print(params)

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
            print(('[{:2d}/{:d}] {:2.0f}m {:2.1f}s - Train Loss: {:.4f} Acc: {:.4f}% - Test Loss: {:.4f} Acc: {:.4f}%').format(
                epoch+1, epochs,
                time_elapsed // 60, time_elapsed % 60,
                tr_loss, tr_acc*100, va_loss, va_acc*100))

            # early stopping
            if tr_acc >= 0.999 and va_acc >= 0.999:
                print(
                    f'Satisfied the accuracy threshold. Abort at {epoch} epoch!')
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

    def _name_handler(self, filename, overwrite):
        arr = filename.split('.')

        # handle wrong extension
        extension = 'pt'

        if len(arr) > 1 and arr[-1] != extension:
            arr[len(arr)-1] = extension
        elif len(arr) == 1:
            arr.append(extension)
        filename = '.'.join(arr)

        # handle existing file
        if not overwrite and os.path.exists(filename):
            arr = filename.split('.')
            time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            arr.insert(-1, time_str)  # already fixed extension
            print('File {:s} already exists. Save new file as "{:s}"'.format(
                filename, '.'.join(arr)))

        return '.'.join(arr)
