import logging
import time

import numpy as np
import sklearn.neighbors as knn
import torch
from torch.utils.data import DataLoader

from ..datasets import NumeralDataset
from .DefenceContainer import DefenceContainer

logger = logging.getLogger(__name__)


class ApplicabilityDomainContainer(DefenceContainer):
    def __init__(self, model_container, k1=9, k2=12, confidence=0.8):
        super(ApplicabilityDomainContainer, self).__init__(model_container)

        params_received = {
            'k1': k1,
            'k2': k2,
            'confidence': confidence}
        self.defence_params.update(params_received)
        self.device = model_container.device

    def fit(self, hidden_model=None):
        if hidden_model is not None:
            self.hidden_model = hidden_model
        else:
            self.hidden_model = self.dummy_model

        # TODO: implement Stage 1
        logger.warning('Stage 1 of Applicability Domain is NOT impletmented!')
        self._log_time_start()

        self._fit_stage1()
        # self._fit_stage2()
        # self._fit_stage3()

        self._log_time_end('train AD')
        return True

    def defence(self, adv, **kwargs):
        n = len(adv)
        # 1: passed test, 0: blocked by AD
        passed = np.ones(n, dtype=np.int8)
        passed = self._def_state1(adv, passed)
        passed = self._def_state2(adv, passed)
        passed = self._def_state3(adv, passed)
        passed_indices = np.nonzero(passed)
        blocked_indices = np.nonzero(np.zeros_like(passed) - passed)
        return adv[passed_indices], blocked_indices

    def _preprocessing(self, x_np):
        dataset = NumeralDataset(torch.as_tensor(x_np))
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0)

        # run 1 sample to get size of output
        x, _ = next(iter(dataloader))
        x = x.to(self.device)
        outputs = self.hidden_model(x[:1])
        num_components = outputs.size()[1]  # number of hidden components

        x_encoded = torch.zeros(len(x_np), num_components)

        start = 0
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                batch_size = len(x)
                x_out = self.hidden_model(x).view(batch_size, -1)  # flatten
                x_encoded[start: start+batch_size] = x_out
        return x_encoded.cpu().detach().numpy()

    def _fit_stage1(self):
        # Step 1: compute hidden layer outputs from inputs
        dc = self.model_container.data_container
        x_train_np = dc.data_train_np
        self.encode_train_np = self._preprocessing(x_train_np)

        # other parameters for AD
        self.y_train_np = dc.label_train_np
        self.num_components = self.encode_train_np.shape[1]
        self.num_classes = self.model_container.data_container.num_classes

    def _fit_stage2(self):
        self._knn_models = []
        self._k_means = np.zeros(self.num_classes, dtype=np.float32)
        self._k_stds = np.zeros_like(self._k_means)

        self._log_time_start
        # TODO: add KNN models here!
        self._log_time_end(f'train {self.num_classes} KNN models')

    def _fit_stage3(self):
        pass

    def _def_state1(self, adv, passed):
        return passed

    def _def_state2(self, adv, passed):
        return passed

    def _def_state3(self, adv, passed):
        return passed

    @staticmethod
    def dummy_model(x):
        """
        Return the input. Do nothing. Use this method when we don't need a hidden 
        layer encoding.
        """
        return x
