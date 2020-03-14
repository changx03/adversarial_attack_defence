import sys

import numpy as np
import torch

from attacks import AttackContainer
from basemodels import BCNN, IrisNN, MnistCnnCW, TorchModelContainer
from datasets import DATASET_LIST, DataContainer
from defences import DefenceContainer


def main():
    DATA_ROOT = 'data'
    BATCH_SIZE = 64

    # image datasets: {'MNIST', 'CIFAR10', 'SVHN'}
    # quantitative datasets: {'BankNote', 'BreastCancerWisconsin', 'HTRU2', 'Iris', 'WheatSeed'}
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

    mc = TorchModelContainer(model, dc)
    mc.fit(epochs=5, batch_size=BATCH_SIZE)  # for image
    # mc.fit(epochs=200, batch_size=BATCH_SIZE)
    print('Test acc:', mc.accuracy_test)

    # shouldn't lose results when train multiple times
    mc.fit(epochs=5, batch_size=BATCH_SIZE)
    print('Test acc:', mc.accuracy_test)

    n_test = len(dc.data_test_np)
    indices = np.random.choice(range(n_test), size=int(n_test * .6))
    x = dc.data_test_np[indices]
    print(f'Testing accuracy on {len(x)} samples')
    y = dc.label_test_np[indices]
    accuracy = mc.evaluate(x, y)
    print(f'Accuracy on random test set: {accuracy*100:.4f}%')

    x = np.random.randn(*x[:5].shape)
    score = mc.score(x)
    print('score:', score)

    pred, score = mc.predict(x, require_score=True)
    print('pred:', pred)
    print('score:', score)

    pred, score = mc.predict_one(x[0], require_score=True)
    print('pred one:', pred)
    print('score:', score)

    # mc.save(f'{NAME}-{model_name}')
    # mc.load(f'{NAME}-{model_name}.pt')


if __name__ == '__main__':
    print(sys.version)
    print(*sys.path, sep='\n')

    main()
