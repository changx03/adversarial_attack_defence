"""
This module implements the Feature Squeezing defence.
"""
import logging

from scipy.ndimage import median_filter
import numpy as np

from .detector_container import DetectorContainer
from ..utils import scale_normalize

logger = logging.getLogger(__name__)


class FeatureSqueezing(DetectorContainer):
    """
    Class preforms Feature Squeezing defence
    """

    def __init__(self,
                 model_container,
                 clip_values=None,
                 bit_depth=8):
        """
        Create a FeatureSqueezing class instance.

        :param model_container: Pre-trained classification model.
        :type model_container: `ModelContainerPT`
        :param clip_values: In form `(min, max)`, indicates the minimum and maximum values for each feature.
        :type clip_values: `tuple`
        :param bit_depth: The image color depth for input images. Pass `None` for numeral data.
        :type bit_depth: `int` or `None`
        """
        super(FeatureSqueezing, self).__init__(model_container)

        if clip_values is None:
            dc = model_container.data_container
            clip_values = dc.data_range

        self._params = {
            'clip_values': clip_values,
            'bit_depth': bit_depth,
        }

    def fit(self):
        """
        Train the classifier with noisy inputs.
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
