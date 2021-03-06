import os
import sys

import numpy as np

from aad.basemodels import MnistCnnCW, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer


def main():
    DATA_ROOT = 'data'
    BATCH_SIZE = 128

    # image datasets: {'MNIST', 'CIFAR10', 'SVHN'}
    # numeric datasets: {'BankNote', 'BreastCancerWisconsin', 'HTRU2', 'Iris', 'WheatSeed'}
    NAME = 'MNIST'
    print(f'Starting {NAME} data container...')
    print(DATASET_LIST[NAME])

    dc = DataContainer(DATASET_LIST[NAME], DATA_ROOT)
    dc(size_train=0.8, normalize=True)

    num_features = dc.dim_data[0]
    num_classes = dc.num_classes
    print('Features:', num_features)
    print('Classes:', num_classes)

    ## model in {BCNN, IrisNN, MnistCnnCW}
    model = MnistCnnCW()
    # model = BCNN(num_features, num_classes)
    # model = IrisNN(num_features, num_classes, hidden_nodes=16)  # for Iris
    # model = IrisNN(num_features, num_classes, hidden_nodes=64)
    model_name = model.__class__.__name__
    print('Using model:', model_name)

    print('Expected initial loss: {}'.format(np.log(10)))

    mc = ModelContainerPT(model, dc)
    mc.fit(max_epochs=10, batch_size=BATCH_SIZE)  # for image
    # mc.fit(max_epochs=200, batch_size=BATCH_SIZE)
    print('Test acc:', mc.accuracy_test)

    mc.save(f'{NAME}-{model_name}')
    mc.load(os.path.join('save', f'{NAME}-{model_name}.pt'))

    acc = mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on random test set: {acc*100:.4f}%')


if __name__ == '__main__':
    print(sys.version)
    print(*sys.path, sep='\n')

    main()
