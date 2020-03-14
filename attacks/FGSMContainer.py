import time

import numpy as np
import torch

from .AttackContainer import AttackContainer

FGSM_PARAMS = ['norm', 'eps', 'eps_step', 'targeted', 'num_random_init',
               'batch_size', 'minimal']


class FGSMContainer(AttackContainer):
    attack_params = AttackContainer.attack_params + FGSM_PARAMS

    def __init__(self, model_containter, norm=1e9, eps=.3, eps_step=0.1,
                 targeted=False, num_random_init=0, batch_size=1, minimal=False):
        super(FGSMContainer, self).__init__(model_containter)

        kwargs = {'norm': norm, 'eps': eps, 'eps_step': eps_step,
                  'targeted': targeted, 'num_random_init': num_random_init,
                  'batch_size': batch_size, 'minimal': minimal}
        FGSMContainer.set_params(self, **kwargs)

    def generate(self, count=10, use_test=True, x=None, y=None, **kwargs):
        dc = self.model_container.data_container
        x = np.copy(dc.data_test_np[:count])
        y = np.copy(dc.label_test_np[:count])
        # TODO: implement this!
        return x
