import abc
import logging

logger = logging.getLogger(__name__)


class DefenceContainer(abc.ABC):
    defence_params = dict()

    def __init__(self, model_container):
        assert isinstance(model_container)
        self.model_container = model_container

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.defence_params.keys():
                self.defence_params[key] = value

    @abc.abstractmethod
    def fit(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def defence(self, adv, **kwargs):
        raise NotImplementedError
