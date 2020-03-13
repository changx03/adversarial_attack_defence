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
    BATCH_SIZE = 128
    # TYPE in {'image', 'quantitative'}
    TYPE = 'quantitative'  

    # image in {'MNIST', 'CIFAR10', 'SVHN'}
    # quantitative in {'BankNote', 'BreastCancerWisconsin', 'HTRU2', 'Iris', 'WheatSeed'}
    NAME = 'BankNote'
    print(f'Starting {NAME} data container...')
    IMAGE_DATASET = DATASET_LIST[TYPE][NAME]
    print(IMAGE_DATASET)

    dc = DataContainer(IMAGE_DATASET, DATA_ROOT)
    dc(BATCH_SIZE, normalize=True)

    num_features = dc.dim_data[0]
    num_classes = dc.num_classes
    print('Features:', num_features)
    print('Classes:', num_classes)

    # model in {BCNN, MnistCnnCW}
    model = BCNN(num_features, num_classes)
    model_name = model.__class__.__name__
    print('Using model:', model_name)

    mc = TorchModelContainer(model, dc)
    mc.fit(epochs=200)

    mc.save(f'{NAME}-{model_name}')

    mc.load(f'{NAME}-{model_name}.pt')

if __name__ == '__main__':
    print(sys.version)
    print(*sys.path, sep='\n')

    main()