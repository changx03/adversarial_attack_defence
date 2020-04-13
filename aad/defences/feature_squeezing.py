"""
This module implements the Feature Squeezing defence.
"""
import copy
import logging

import numpy as np
import torch.nn as nn
from scipy.ndimage import median_filter

from ..utils import scale_normalize
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class FeatureSqueezing(DetectorContainer):
    """
    Class preforms Feature Squeezing defence
    """

    def __init__(self,
                 model_container,
                 clip_values=None,
                 smoothing_methods=['median'],
                 bit_depth=None,
                 sigma=None,
                 pretrained=True):
        """
        Create a FeatureSqueezing class instance.

        Parameters
        ----------
        model_container : ModelContainerPT
            Pre-trained classification model.
        smoothing_methods : list of ('median', 'normal', 'binary')
            Select one or more smoothing methods. 'median' filter can only use on images. 'binary' filter maps features
            based on `bit_depth`. 'normal' filter generates noise based on normal distribution.
        clip_values : tuple of (min, max)
            Indicates the minimum and maximum values of each feature. If the data is normalized, it should be (0.0, 1.0).
        bit_depth : int, optional
            The image color depth for input images. Pass `None` for numeral data. Required for 'binary' filter. e.g.: 8
        sigma : float, optional
            The Standard Deviation of normal distribution. Required for 'normal' filter. e.g.: 0.1
        pretrained : bool
            Load the pre-trained parameters before train the smoothing models.
        """
        super(FeatureSqueezing, self).__init__(model_container)

        if clip_values is None:
            dc = model_container.data_container
            clip_values = dc.data_range

        self._params = {
            'clip_values': clip_values,
            'bit_depth': bit_depth,
            'sigma': sigma,
            'pretrained': pretrained,
        }
        self._smoothing_methods = smoothing_methods

        if 'binary' in smoothing_methods and bit_depth is None:
            raise ValueError('`bit_depth` cannot be `None` for binary filter.')
        if 'normal' in smoothing_methods and sigma is None:
            raise ValueError('`sigma` cannot be `None` for normal filter.')

        mc = self.model_container
        self._base_model = copy.deepcopy(mc.model)
        if not pretrained:
            for param in self._base_model.parameters():
                nn.init.uniform_(param, a=-1.0, b=1.0)

        self.models = []
        for method in smoothing_methods:
            self.models.append({
                'model_container': None,
                'method': method
            })

    @property
    def smoothing_methods(self):
        return self._smoothing_methods

    def fit(self):
        """
        Train the smoothing models with selected filters.
        """
        mc = self.model_container
        dc = mc.data_container
        x = dc.x_train
        y = dc.y_train
        clip_values = self._params['clip_values']
        bit_depth = self._params['bit_depth']

        x_norm = scale_normalize(x, clip_values[0], clip_values[1])
        max_val = np.rint(2 ** bit_depth - 1)
        res = np.rint(x_norm * max_val) / max_val
        res = res * (clip_values[1] - clip_values[0])
        res += clip_values[0]
        print('ph')

    def save(self, filename, overwrite=False):
        """Save trained parameters."""
        raise NotImplementedError

    def load(self, filename):
        """Load pre-trained parameters."""
        raise NotImplementedError

    def detect(self, adv, pred, return_passed_x):
        """
        Compare the predictions between adv. training model and original model, 
        and block all unmatched results.
        """
        raise NotImplementedError
