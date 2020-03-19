import time

import numpy as np
import torch
from art.attacks import SaliencyMapMethod
from art.classifiers import PyTorchClassifier

from ..utils import swap_image_channel
from .AttackContainer import AttackContainer


class JacobianSaliencyContainer(AttackContainer):
    def __init__(self, model_container, theta=0.1, gamma=1.0, batch_size=16):
        super(JacobianSaliencyContainer, self).__init__(model_container)

        dim_data = model_container.data_container.dim_data
        assert len(dim_data) == 3, \
            'Jacobian Saliency Map attack only works on images'

        params_received = {
            'theta': theta,
            'gamma': gamma,
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
        print(clip_values)
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

        # handle (h, w, c) to (c, h, w)
        data_type = self.model_container.data_container.type
        if data_type == 'image' and x.shape[1] not in (1, 3):
            x = swap_image_channel(x)

        # handle the situation where targets are more than test set
        if targets is not None:
            assert len(targets) >= len(x)
            targets = targets[:len(x)]  # trancate targets

        attack = SaliencyMapMethod(
            classifier=self.classifier, **self.attack_params)

        # predict the outcomes
        adv = attack.generate(x, y=targets)
        y_adv, y_clean = self.predict(adv, x)

        time_elapsed = time.time() - since
        print('Time to complete training {} adversarial examples: {:2.0f}m {:2.1f}s'.format(
            count, time_elapsed // 60, time_elapsed % 60))
        return adv, y_adv, np.copy(x), y_clean
