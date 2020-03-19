import logging

import numpy as np
import sklearn.neighbors as knn
import torch
from torch.utils.data import DataLoader

from ..datasets import NumeralDataset
from .DefenceContainer import DefenceContainer

logger = logging.getLogger(__name__)


class ApplicabilityDomainContainer(DefenceContainer):
    def __init__(self, model_container, hidden_model=None, k1=9, k2=12, confidence=0.8):
        super(ApplicabilityDomainContainer, self).__init__(model_container)

        params_received = {
            'k1': k1,
            'k2': k2,
            'confidence': confidence}
        self.params.update(params_received)
        self.device = model_container.device
        self.num_classes = model_container.data_container.num_classes

        if hidden_model is not None:
            self.hidden_model = hidden_model
        else:
            self.hidden_model = self.dummy_model

        # objects used for defence
        self.num_components = 0  # We don't know the value yet
        self._knn_models = []
        self.k_means = np.zeros(self.num_classes, dtype=np.float32)
        self.k_stds = np.zeros_like(self.k_means)
        self.thresholds = np.zeros(self.num_classes, dtype=np.float32)
        self.encode_train_np = None
        self.y_train_np = None

    def fit(self):
        self._log_time_start()

        # Step 1: compute hidden layer outputs from inputs
        dc = self.model_container.data_container
        x_train_np = dc.data_train_np
        self.encode_train_np = self._preprocessing(x_train_np)
        self.y_train_np = dc.label_train_np
        self.num_components = self.encode_train_np.shape[1]

        self._fit_stage1()
        self._fit_stage2()
        # self._fit_stage3()

        self._log_time_end('train AD')
        return True

    def defence(self, adv):
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
        # TODO: implement Stage 1
        logger.warning('Stage 1 of Applicability Domain is NOT impletmented!')

    def _fit_stage2(self):
        self._log_time_start()

        k1 = self.params['k1']
        zeta = self.params['confidence']
        for l in range(self.num_classes):
            indices = np.where(self.y_train_np == 1)
            x = self.encode_train_np[indices]
            model = knn.KNeighborsClassifier(n_neighbors=k1, n_jobs=-1)
            model.fit(x, np.ones(len(x)))
            self._knn_models.append(model)

            dist, _ = model.kneighbors(x, n_neighbors=k1+1)
            avg_dist = np.sum(dist, axis=1) / float(k1)
            self.k_means[l] = np.mean(avg_dist)
            self.k_stds[l] = np.std(avg_dist)
            self.thresholds[l] = self.k_means[l] + zeta * self.k_stds[l]

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
