import time

import numpy as np
import torch
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer


class FGSMContainer(AttackContainer):
    def __init__(self, model_container, norm=np.inf, eps=.3, eps_step=0.1,
                 targeted=False, num_random_init=0, batch_size=64, minimal=False):
        '''
        Fast Gradient Sign Method. Use L-inf norm as default
        '''
        super(FGSMContainer, self).__init__(model_container)

        params_received = {
            'norm': norm,
            'eps': eps,
            'eps_step': eps_step,
            'targeted': targeted,
            'num_random_init': num_random_init,
            'batch_size': batch_size,
            'minimal': minimal}
        self.attack_params.update(params_received)

        # use IBM ART pytorch module wrapper
        # the model used here should be already trained
        model = self.model_container.model
        loss_fn = self.model_container.model.loss_fn
        clip_values = self.model_container.data_container.data_range
        optimizer = self.model_container.model.optimizer
        num_classes = self.model_container.data_container.num_classes
        dim_data = self.model_container.data_container.dim_data
        self.classifier = PyTorchClassifier(
            model=model,
            clip_values=clip_values,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=dim_data,
            nb_classes=num_classes)

    def generate(self, count=1000, use_test=True, x=None, **kwargs):
        assert use_test or x is not None

        since = time.time()
        # parameters should able to set before training
        self.set_params(**kwargs)
        dc = self.model_container.data_container
        if len(dc.data_test_np) < count:
            count = len(dc.data_test_np)

        x = np.copy(dc.data_test_np[:count]) if use_test else np.copy(x)

        self.set_params(**kwargs)
        attack = FastGradientMethod(self.classifier, **self.attack_params)

        # predict the outcomes
        adv = attack.generate(x)
        y_adv, y_clean = self.predict(adv, x)

        time_elapsed = time.time() - since
        print('Time taken for training {} adversarial examples: {:2.0f}m {:2.1f}s'.format(
            count, time_elapsed // 60, time_elapsed % 60))
        return adv, y_adv, np.copy(x), y_clean
