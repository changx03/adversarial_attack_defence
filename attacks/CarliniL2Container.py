import time

import numpy as np
import torch
from art.attacks import CarliniL2Method
from art.classifiers import PyTorchClassifier

from .AttackContainer import AttackContainer


class CarliniL2Container(AttackContainer):
    def __init__(self, model_container, confidence=0.0, targeted=False,
                 learning_rate=1e-2, binary_search_steps=10, max_iter=100,
                 initial_const=1e-2, max_halving=5, max_doubling=10, batch_size=8):
        super(CarliniL2Container, self).__init__(model_container)

        params_received = {
            'confidence': confidence,
            'targeted': targeted,
            'learning_rate': learning_rate,
            'binary_search_steps': binary_search_steps,
            'max_iter': max_iter,
            'initial_const': initial_const,
            'max_halving': max_halving,
            'max_doubling': max_doubling,
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

    def generate(self, count=1000, use_testset=True, x=None, targets=None, **kwargs):
        assert use_testset or x is not None

        since = time.time()
        # parameters should able to set before training
        self.set_params(**kwargs)

        dc = self.model_container.data_container
        # handle the situation where testset has less samples than we want
        if use_testset and len(dc.data_test_np) < count:
            count = len(dc.data_test_np)

        x = np.copy(dc.data_test_np[:count]) if use_testset else np.copy(x)

        targeted = targets is not None
        # handle the situation where targets are more than test set
        if targets is not None:
            assert len(targets) >= len(x)
            targets = targets[:len(x)]  # trancate targets

        self.attack_params['targeted'] = targeted
        attack = CarliniL2Method(
            classifier=self.classifier, **self.attack_params)

        # predict the outcomes
        adv = attack.generate(x, targets)
        y_adv, y_clean = self.predict(adv, x)

        time_elapsed = time.time() - since
        print('Time taken for training {} adversarial examples: {:2.0f}m {:2.1f}s'.format(
            count, time_elapsed // 60, time_elapsed % 60))
        return adv, y_adv, np.copy(x), y_clean