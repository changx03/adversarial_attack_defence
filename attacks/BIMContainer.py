import time

import numpy as np
import torch
from art.attacks import BasicIterativeMethod
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer


class BIMContainer(AttackContainer):
    def __init__(self, model_container, eps=.3, eps_step=0.1, max_iter=100,
                 targeted=False, batch_size=64):
        super(BIMContainer, self).__init__(model_container)

        params_received = {
            'eps': eps,
            'eps_step': eps_step,
            'max_iter': max_iter,
            'targeted': targeted,
            'batch_size': batch_size}
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

    def generate(self, count=1000, use_test=True, x=None, targets=None, **kwargs):
        assert use_test or x is not None

        since = time.time()
        # parameters should able to set before training
        self.set_params(**kwargs)
        dc = self.model_container.data_container
        if use_test:
            if len(dc.data_test_np) < count:
                count = len(dc.data_test_np)
        else:
            assert x is not None
            count = len(x)

        x = np.copy(dc.data_test_np[:count]) if use_test else np.copy(x)
        assert targets == None or len(targets) == len(x)

        self.set_params(**kwargs)
        attack = BasicIterativeMethod(classifier=self.classifier, **kwargs)

        # predict the outcomes
        adv = attack.generate(x, targets)
        y_adv, y_clean = self.predict(adv, x)

        time_elapsed = time.time() - since
        print('Time taken for training {} adversarial examples: {:2.0f}m {:2.1f}s'.format(
            count, time_elapsed // 60, time_elapsed % 60))
        return adv, y_adv, np.copy(x), y_clean
