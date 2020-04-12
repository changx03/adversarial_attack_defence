"""
This module implements the Fast Gradient Sign Method attack.
"""
import logging
import time

import numpy as np
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier

from ..utils import swap_image_channel
from .attack_container import AttackContainer

logger = logging.getLogger(__name__)


class FGSMContainer(AttackContainer):
    def __init__(self, model_container, norm=np.inf, eps=.3, eps_step=0.1,
                 targeted=False, num_random_init=0, batch_size=64, minimal=False):
        """
        Fast Gradient Sign Method. Use L-inf norm as default
        """
        super(FGSMContainer, self).__init__(model_container)

        params_received = {
            'norm': norm,
            'eps': eps,
            'eps_step': eps_step,
            'targeted': targeted,
            'num_random_init': num_random_init,
            'batch_size': batch_size,
            'minimal': minimal}
        self.attack_params = params_received

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

    def generate(self, count=1000, use_testset=True, x=None, **kwargs):
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

        # predict the outcomes
        adv = self._generate(xx)
        y_adv, y_clean = self.predict(adv, xx)

        # ensure the outputs and inputs have same shape
        if x.shape != adv.shape:
            adv = swap_image_channel(adv)
        time_elapsed = time.time() - since
        logger.info('Time to complete training %d adv. examples: %dm %.3fs',
                    count, int(time_elapsed // 60), time_elapsed % 60)
        return adv, y_adv, np.copy(x), y_clean

    def _generate(self, x):
        attack = FastGradientMethod(self.classifier, **self.attack_params)
        return attack.generate(x)
