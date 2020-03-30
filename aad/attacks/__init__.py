from .attack_container import AttackContainer
from .bim_attack import BIMContainer
from .carlini_l2_attack import CarliniL2Container
from .deepfool_attack import DeepFoolContainer
from .dummy_attack import DummyAttack
from .fgsm_attack import FGSMContainer
from .saliency_map_attack import SaliencyContainer
from .zoo_attack import ZooContainer


def get_attack(name):
    """Returns a attack based on the given name."""
    if name == 'FGSM':
        return FGSMContainer
    elif name == 'BIM':
        return BIMContainer
    elif name == 'Carlini':
        return CarliniL2Container
    elif name == 'DeepFool':
        return DeepFoolContainer
    elif name == 'Saliency':
        return SaliencyContainer
    else:
        raise AttributeError('Received unknown attack "{}"'.format(name))