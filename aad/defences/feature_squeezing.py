"""
This module implements the Feature Squeezing defence.
"""
import logging

from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class FeatureSqueezing(DetectorContainer):
    """
    Class preforms Feature Squeezing defence
    """

    def __init__(self,
                 model_container,
                 clip_values,
                 bit_depth):
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

        self._params = {
            'clip_values': clip_values,
            'bit_depth': bit_depth,
        }

    def fit(self):
        """
        Train the classifier with noisy inputs.
        """
        raise NotImplementedError

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
