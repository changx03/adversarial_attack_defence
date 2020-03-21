import logging
import os
import sys

import numpy as np
import torch

from aad.attacks import ZooContainer
from aad.basemodels import MnistCnnCW, TorchModelContainer
from aad.datasets import DATASET_LIST, DataContainer

logger = logging.getLogger(__name__)

DATA_ROOT = 'data'
BATCH_SIZE = 64
EPOCHS = 30
NAME = 'MNIST'


def main():
    print(f'Starting {NAME} data container...')
    print(DATASET_LIST[NAME])

    # Step 1: select dataset
    dc = DataContainer(DATASET_LIST[NAME], DATA_ROOT)
    dc(size_train=0.8, normalize=True)

    num_features = dc.dim_data[0]
    num_classes = dc.num_classes
    print('Features:', num_features)
    print('Classes:', num_classes)

    # Step 2: train model
    model = MnistCnnCW()
    model_name = model.__class__.__name__
    print('Using model:', model_name)

    mc = TorchModelContainer(model, dc)

    model_filename = f'example-mnist-e{EPOCHS}.pt'
    file_path = os.path.join('save', model_filename)

    if not os.path.exists(file_path):
        mc.fit(epochs=EPOCHS, batch_size=BATCH_SIZE)
        mc.save(model_filename, overwrite=True)
    else:
        print('Found saved model!')
        mc.load(file_path)

    acc = mc.evaluate(dc.data_test_np, dc.label_test_np)
    print(f'Accuracy on random test set: {acc*100:.4f}%')

    # Step 3: attack the model
    # TODO: Unable to generate adversarial examples successfully!
    attack = ZooContainer(
        mc,
        targeted=False,
        learning_rate=1e-2,
        max_iter=10,
        binary_search_steps=3,
        abort_early=False,
        use_resize=False,
        use_importance=False)

    adv, y_adv, x_clean, y_clean = attack.generate(count=30)
    accuracy = mc.evaluate(adv, y_clean)
    print('Accuracy on adversarial examples: {:.4f}%'.format(
        accuracy*100))

    attack.save_attack(
        f'example-mnist-e{EPOCHS}', adv, y_adv, x_clean, y_clean)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
