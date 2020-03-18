import logging
import time

import torch
from torch.utils.data import DataLoader

from .DefenceContainer import DefenceContainer
from ..datasets import NumeralDataset

logger = logging.getLogger(__name__)


class ApplicabilityDomainContainer(DefenceContainer):
    def __init__(self, model_container, k1=9, k2=12, confidence=0.8):
        super(ApplicabilityDomainContainer, self).__init__(model_container)

        params_received = {
            'k1': k1,
            'k2': k2,
            'confidence': confidence}
        self.defence_params.update(params_received)

    def fit(self, hidden_model=None):
        self.hidden_model = hidden_model
        # TODO: implement Stage 1
        logger.warning('Stage 1 for Applicability Domain is NOT impletmented!')
        self._log_time_start()

        self._fit_stage1()
        self._fit_stage2()
        self._fit_stage3()

        self._log_time_end('train AD')
        return True

    def defence(self, adv, **kwargs):
        pass

    def _preprocessing(self, x_np, y_np):
        dataset = NumeralDataset(
            torch.as_tensor(x_np),
            torch.as_tensor(y_np))
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0)
        
        # TODO: this returns encoded ndarray
        return x_np, y_np

    def _fit_stage1(self):
        # Step 1: compute hidden layer outputs from inputs
        dc = self.model_container.data_container
        device = self.model_container.device
        loader_train = dc.get_dataloader(
            batch_size=64, is_train=True, shuffle=False)
        # run 1 sample to get size of output
        x, _ = next(iter(loader_train))
        outputs = self.hidden_model(x[:1])
        self.num_components = outputs.size()[1]  # number of hidden components
        self.num_train = len(loader_train.dataset)
        self._encoded_train = torch.empty(self.num_train, self.num_components)
        self._y_train = torch.empty(self.num_train, dtype=torch.long)

        start = 0
        with torch.no_grad():
            for x, y in loader_train:
                x = x.to(device)
                y = y.to(device)
                batch_size = len(x)
                x_encoded = self.hidden_model(
                    x).view(batch_size, -1)  # flatten
                self._encoded_train[start: start+batch_size] = x_encoded
                self._y_train[start: start+batch_size] = y

    def _fit_stage2(self):
        pass

    def _fit_stage3(self):
        pass

    def _def_state1(self, adv):
        pass

    def _def_state2(self):
        pass

    def _def_state3(self):
        pass
