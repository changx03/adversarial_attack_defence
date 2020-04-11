"""
This module implements the base class for a detector
"""
import abc
import logging
import time

from ..basemodels import ModelContainerPT

logger = logging.getLogger(__name__)


class DetectorContainer(abc.ABC):
    """
    Class performing adversarial detection
    """
    params = dict()  # Override this in child class

    def __init__(self, model_container):
        assert isinstance(model_container, ModelContainerPT)
        self.model_container = model_container
        self._since = 0.0

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params.keys():
                self.params[key] = value

    @abc.abstractmethod
    def fit(self):
        """Train the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, filename, overwrite=False):
        """Save the defence"""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, filename):
        """Load the defence"""
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self, adv, pred):
        """Detect adversarial examples."""
        raise NotImplementedError

    def _log_time_start(self):
        self._since = time.time()

    def _log_time_end(self, title=None):
        time_elapsed = time.time() - self._since
        title = ' [' + title + ']' if title else ''
        logger.debug(
            'Time to complete%s: %dm %.3fs',
            title, int(time_elapsed // 60), time_elapsed % 60)
