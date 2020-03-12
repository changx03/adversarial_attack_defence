import os
import time

import numpy as np
import torch
import torch.nn as nn


class ModelContainer:
    def __init__(self, model, data_container):
        self.model = model
        self.data_container = data_container
        assert isinstance(model, nn.Module)

    def fit(self, **kwargs):
        since = time.time()
        if isinstance(self.model, nn.Module):
            assert 'epochs' in kwargs, "Argument 'epochs' is required"
            epochs = kwargs['epochs']

            self._fit_torch(epochs)
        else:
            # TODO: methods other than CNN
            raise Exception('Not implemented!')
        time_elapsed = time.time() - since
        print('Time taken for training: {:2.0f}m {:2.1f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def save(self):
        return None

    def load(self):
        return None

    def pred(self, x):
        return x

    def pred_one(self, x):
        return x

    def _fit_torch(self, epochs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # Not all models run multiple iterations
        self.loss_train = np.zeros(epochs, dtype=np.float32)
        self.loss_test = np.zeros(epochs, dtype=np.float32)
        self.accuracy_train = np.zeros(epochs, dtype=np.float32)
        self.accuracy_test = np.zeros(epochs, dtype=np.float32)

        self.model.to(device)
        lr = self.model.lr
        momentum = self.model.momentum
        optimizer = self.model.optimizer(
            self.model.parameters(), lr=lr, momentum=momentum)
        
        for epoch in range(epochs):
            time_start = time.time()

            tr_loss, tr_acc = self._train_torch(optimizer, device)
            va_loss, va_acc = self._validate_torch(device)

            time_elapsed = time.time() - time_start
            print(('[{:2d}/{:d}] {:2.0f}m {:2.1f}s Train Loss: {:.4f} Acc: {:.4f}% - Test Loss: {:.4f} Acc: {:.4f}%').format(
                epoch+1, epochs,
                time_elapsed // 60, time_elapsed % 60,
                tr_loss, tr_acc*100.,
                va_loss, va_acc*100.))

    def _train_torch(self, optimizer, device):
        self.model.train()
        total_loss = 0.
        corrects = 0.
        loader = self.data_container.dataloader_train
        loss_fn = self.model.loss_fn

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            optimizer.zero_grad()
            output = self.model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            # for logging
            total_loss += loss.item() * batch_size
            preds = output.max(1, keepdim=True)[1]
            corrects += preds.eq(y.view_as(preds)).sum().item()

        n = self.data_container.num_train
        total_loss = total_loss / n
        acc = corrects / n
        return total_loss, acc

    def _validate_torch(self, device):
        self.model.eval()
        total_loss = 0.
        corrects = 0
        loader = self.data_container.dataloader_test
        loss_fn = self.model.loss_fn

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                batch_size = x.size(0)
                output = self.model(x)
                loss = loss_fn(output, y)
                total_loss += loss.item() * batch_size
                preds = output.max(1, keepdim=True)[1]
                corrects += preds.eq(y.view_as(preds)).sum().item()

        n = self.data_container.num_test
        total_loss = total_loss / n
        accuracy = corrects / n
        return total_loss, accuracy
