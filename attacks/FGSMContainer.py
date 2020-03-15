import time

import numpy as np
import torch
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer

FGSM_PARAMS = ['norm', 'eps', 'eps_step', 'targeted', 'num_random_init',
               'batch_size', 'minimal']


class FGSMContainer(AttackContainer):
    attack_params = AttackContainer.attack_params + FGSM_PARAMS

    def __init__(self, model_container, norm=1e9, eps=.3, eps_step=0.1,
                 targeted=False, num_random_init=0, batch_size=1, minimal=False):
        super(FGSMContainer, self).__init__(model_container)

        kwargs = {'norm': norm, 'eps': eps, 'eps_step': eps_step,
                  'targeted': targeted, 'num_random_init': num_random_init,
                  'batch_size': batch_size, 'minimal': minimal}
        self.set_params(**kwargs)

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

    def generate(self, count=1000, use_test=True, x=None, labels=None, **kwargs):
        assert use_test or x is not None
        assert use_test or labels is not None
        
        since = time.time()
        # parameters should able to set before training
        self.set_params(**kwargs)
        dc = self.model_container.data_container
        if len(dc.data_test_np) < count:
            count = len(dc.data_test_np)

        x = np.copy(dc.data_test_np[:count]) if use_test else np.copy(x)
        labels = np.copy(dc.label_test_np[:count]) if use_test else np.copy(labels)

        attack = FastGradientMethod(
            classifier=self.classifier,
            norm=np.inf,
            eps=.3,
            eps_step=0.1,
            targeted=False,
            num_random_init=0,
            batch_size=1,
            minimal=False)

        # predict the outcomes
        adv = attack.generate(x)
        y_adv = self.model_container.predict(adv)
        x_clean = np.copy(x)
        y_clean = labels

        time_elapsed = time.time() - since
        print('Time taken for training {} adversarial examples: {:2.0f}m {:2.1f}s'.format(
            count, time_elapsed // 60, time_elapsed % 60))
        return adv, y_adv, x_clean, y_clean
