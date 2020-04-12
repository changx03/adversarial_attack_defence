"""
This module implements the Carlini and Wagner L2 attack.
"""
import logging
import time

import numpy as np
from art.attacks import CarliniL2Method
from art.classifiers import PyTorchClassifier

from ..utils import get_range, swap_image_channel
from .attack_container import AttackContainer

logger = logging.getLogger(__name__)


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
        self.attack_params = params_received

        # use IBM ART pytorch module wrapper
        # the model used here should be already trained
        model = self.model_container.model
        dc = self.model_container.data_container
        loss_fn = model.loss_fn
        dc = self.model_container.data_container
        clip_values = get_range(dc.data_train_np, dc.data_type == 'image')
        optimizer = model.optimizer
        num_classes = dc.num_classes
        dim_data = dc.dim_data
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
        data_type = self.model_container.data_container.data_type
        if data_type == 'image' and x.shape[1] not in (1, 3):
            xx = swap_image_channel(x)
        else:
            xx = x

        adv = self._generate(xx)
        y_adv, y_clean = self.predict(adv, xx)

        # ensure the outputs and inputs have same shape
        if x.shape != adv.shape:
            adv = swap_image_channel(adv)
        time_elapsed = time.time() - since
        logger.info('Time to complete training %d adv. examples: %dm %.3fs',
                    count, int(time_elapsed // 60), time_elapsed % 60)
        return adv, y_adv, x, y_clean

    def _generate(self, x, targets=None):
        targeted = targets is not None
        # handle the situation where targets are more than test set
        if targets is not None:
            assert len(targets) >= len(x)
            targets = targets[:len(x)]  # trancate targets

        self.attack_params['targeted'] = targeted
        attack = CarliniL2Method(
            classifier=self.classifier, **self.attack_params)

        # predict the outcomes
        if targets is not None:
            adv = attack.generate(x, targets)
        else:
            adv = attack.generate(x)
        return adv
