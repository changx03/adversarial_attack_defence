"""
This module implements the Jacobian Saliency Map attack.
"""
import logging
import time

import numpy as np
from art.attacks import SaliencyMapMethod
from art.classifiers import PyTorchClassifier

from ..utils import swap_image_channel
from .attack_container import AttackContainer

logger = logging.getLogger(__name__)


class SaliencyContainer(AttackContainer):
    def __init__(self, model_container, theta=0.1, gamma=1.0, batch_size=16):
        super(SaliencyContainer, self).__init__(model_container)

        dim_data = model_container.data_container.dim_data
        assert len(dim_data) == 3, \
            'Jacobian Saliency Map attack only works on images'

        params_received = {
            'theta': theta,
            'gamma': gamma,
            'batch_size': batch_size}
        self.attack_params = params_received

        # use IBM ART pytorch module wrapper
        # the model used here should be already trained
        model = self.model_container.model
        loss_fn = self.model_container.model.loss_fn
        clip_values = self.model_container.data_container.data_range
        optimizer = self.model_container.model.optimizer
        num_classes = self.model_container.data_container.num_classes
        dim_data = self.model_container.data_container.dim_data
        logger.debug('clip_values = %s', str(clip_values))

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

        xx = np.copy(dc.data_test_np[:count]) if use_testset else np.copy(x)

        # handle (h, w, c) to (c, h, w)
        data_type = self.model_container.data_container.data_type
        if data_type == 'image' and xx.shape[1] not in (1, 3):
            xx = swap_image_channel(x)

        adv = self._generate(xx)
        y_adv, y_clean = self.predict(adv, xx)

        # ensure the outputs and inputs have same shape
        if x.shape != adv.shape:
            adv = swap_image_channel(adv)
        time_elapsed = time.time() - since
        logger.info('Time to complete training %d adv. examples: %dm %.3fs',
                    count, int(time_elapsed // 60), time_elapsed % 60)
        return adv, y_adv, np.copy(x), y_clean

    def _generate(self, x, targets=None):
        targeted = targets is not None
        # handle the situation where targets are more than test set
        if targets is not None:
            assert len(targets) >= len(x)
            targets = targets[:len(x)]  # trancate targets

        self.attack_params['targeted'] = targeted
        attack = SaliencyMapMethod(
            classifier=self.classifier, **self.attack_params)

        # predict the outcomes
        if targets is not None:
            adv = attack.generate(x, targets)
        else:
            adv = attack.generate(x)
        return adv
