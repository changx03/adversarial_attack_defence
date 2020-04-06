from .model_bc import BCNN
from .model_cifar import CifarCnn
from .model_container_pt import ModelContainerPT
from .model_iris import IrisNN
from .model_mnist import MnistCnnCW, MnistCnnCW_hidden
from .model_mnist_v2 import MnistCnnV2
from .model_resnet_cifar import CifarResnet50


def get_model(name):
    """Returns a model based on the given name."""
    if name == 'BCNN':
        return BCNN
    elif name == 'CifarCnn':
        return CifarCnn
    elif name == 'CifarResnet':
        return CifarResnet50
    elif name == 'IrisNN':
        return IrisNN
    elif name == 'MnistCnnCW':
        return MnistCnnCW
    elif name == 'MnistCnnV2':
        return MnistCnnV2
    else:
        raise AttributeError('Received unknown model "{}"'.format(name))
