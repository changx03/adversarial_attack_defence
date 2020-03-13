import sys

import numpy as np
import torch

from attacks import AttackContainer
from basemodels import BCNN, IrisNN, MnistCnnCW, TorchModelContainer
from datasets import (DATASET_LIST, DataContainer, get_image_list,
                      get_quantitative_list)
from defences import DefenceContainer


def main():
    DATA_ROOT = 'data'
    BATCH_SIZE = 64
    # TYPE in {'image', 'quantitative'}
    TYPE = 'quantitative'

    # image in {'MNIST', 'CIFAR10', 'SVHN'}
    # quantitative in {'BankNote', 'BreastCancerWisconsin', 'HTRU2', 'Iris', 'WheatSeed'}
    NAME = 'WheatSeed'
    print(f'Starting {NAME} data container...')
    IMAGE_DATASET = DATASET_LIST[TYPE][NAME]
    print(IMAGE_DATASET)

    dc = DataContainer(IMAGE_DATASET, DATA_ROOT)
    dc(BATCH_SIZE, size_train=0.8, normalize=True)

    num_features = dc.dim_data[0]
    num_classes = dc.num_classes
    print('Features:', num_features)
    print('Classes:', num_classes)

    # model in {BCNN, IrisNN, MnistCnnCW}
    # model = BCNN(num_features, num_classes)
    model = IrisNN(num_features, num_classes, hidden_nodes=64)
    model_name = model.__class__.__name__
    print('Using model:', model_name)

    mc = TorchModelContainer(model, dc)
    mc.fit(epochs=200)

    # mc.save(f'{NAME}-{model_name}')
    # mc.load(f'{NAME}-{model_name}.pt')


if __name__ == '__main__':
    print(sys.version)
    print(*sys.path, sep='\n')

    main()
