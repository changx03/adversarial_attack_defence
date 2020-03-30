"""
This module implements the ZOO attack.
"""
import logging
import time

import numpy as np
from art.attacks import ZooAttack
from art.classifiers import PyTorchClassifier

from ..utils import swap_image_channel
from .attack_container import AttackContainer

logger = logging.getLogger(__name__)


class ZooContainer(AttackContainer):
    """
    Zeroth-Order Optimization attack (Zoo) is a black-box attack. This attack 
    is a variant of the Carlini and Wagner attack which uses ADAM coordinate 
    descent to perform numerical estimation of gradients.
    """
    def __init__(
            self,
            model_container,
            confidence=0.0,
            targeted=False,
            learning_rate=1e-2,
            max_iter=10,
            binary_search_steps=1,
            initial_const=1e-3,
            abort_early=True,
            use_resize=True,
            use_importance=True,
            nb_parallel=128,
            batch_size=1,
            variable_h=1e-4):
        super(ZooContainer, self).__init__(model_container)

        params_received = {
            'confidence': confidence,
            'targeted': targeted,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'binary_search_steps': binary_search_steps,
            'initial_const': initial_const,
            'abort_early': abort_early,
            'use_resize': use_resize,
            'use_importance': use_importance,
            'nb_parallel': nb_parallel,
            'batch_size': batch_size,
            'variable_h': variable_h
        }
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
            x = swap_image_channel(x)

        targeted = targets is not None
        # handle the situation where targets are more than test set
        if targets is not None:
            assert len(targets) >= len(x)
            targets = targets[:len(x)]  # trancate targets

        self.attack_params['targeted'] = targeted
        attack = ZooAttack(
            classifier=self.classifier, **self.attack_params)

        # predict the outcomes
        if targets is not None:
            adv = attack.generate(x, targets)
        else:
            adv = attack.generate(x)
        y_adv, y_clean = self.predict(adv, x)

        time_elapsed = time.time() - since
        logger.info('Time to complete training %d adversarial examples: %dm %.3fs',
                    count, int(time_elapsed // 60), time_elapsed % 60)
        return adv, y_adv, np.copy(x), y_clean
