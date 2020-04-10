"""
Module for classification models.
"""
from .model_bc import BCNN
from .model_cifar import CifarCnn
from .model_container_pt import ModelContainerPT
from .model_iris import IrisNN
from .model_mnist import MnistCnnCW, MnistCnnCW_hidden
from .model_mnist_v2 import MnistCnnV2
from .model_resnet_cifar import CifarResnet50

AVALIABLE_MODELS = (
    'BCNN',
    'CifarCnn',
    'CifarResnet50',
    'IrisNN',
    'MnistCnnCW',
    'MnistCnnV2',
)

def get_model(name):
    """Returns a model based on the given name."""
    if name == AVALIABLE_MODELS[0]:
        return BCNN
    elif name == AVALIABLE_MODELS[1]:
        return CifarCnn
    elif name == AVALIABLE_MODELS[2]:
        return CifarResnet50
    elif name == AVALIABLE_MODELS[3]:
        return IrisNN
    elif name == AVALIABLE_MODELS[4]:
        return MnistCnnCW
    elif name == AVALIABLE_MODELS[5]:
        return MnistCnnV2
    else:
        raise AttributeError('Received unknown model "{}"'.format(name))
