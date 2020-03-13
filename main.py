import sys

import torch
import numpy as np

from attacks import AttackContainer
from basemodels import TorchModelContainer, BCNN, MnistCnnCW
from datasets import (DATASET_LIST, DataContainer, get_image_list,
                      get_quantitative_list)
from defences import DefenceContainer

def main():
    DATA_ROOT = 'data'
    BATCH_SIZE = 64
    # TYPE := {'image', 'quantitative'}
    TYPE = 'quantitative'  
    # image: {'MNIST', 'CIFAR10', 'SVHN'}
    # quantitative: {'BankNote', 'BreastCancerWisconsin', 'WheatSeed', 'HTRU2'}
    NAME = 'BreastCancerWisconsin'
    print(f'Starting {NAME} data container...')
    IMAGE_DATASET = DATASET_LIST[TYPE][NAME]
    dc = DataContainer(IMAGE_DATASET, DATA_ROOT)
    dc(BATCH_SIZE, normalize=True)
    # model := {BCNN, MnistCnnCW}
    model = BCNN()
    mc = TorchModelContainer(model, dc)
    mc.fit(epochs=200)

    mc.save('BC_NN2')

    # mc.load('BC_NN1.pt')

if __name__ == '__main__':
    print(sys.version)
    print(*sys.path, sep='\n')

    main()