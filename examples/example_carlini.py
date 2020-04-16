import os
import sys
import logging

import numpy as np

from aad.attacks import CarliniL2V2Container
from aad.basemodels import ModelContainerPT, MnistCnnCW
from aad.utils import get_time_str, master_seed, get_data_path, get_l2_norm
from aad.datasets import DATASET_LIST, DataContainer

logging.basicConfig(level=logging.DEBUG)

DATA_NAME = 'MNIST'
SEED = 4096
MODEL_FILE = os.path.join('save', 'MnistCnnCW_MNIST_e50.pt')


def main():
    master_seed(SEED)

    dataset = DATASET_LIST[DATA_NAME]
    dc = DataContainer(dataset, get_data_path())
    dc()
    model = MnistCnnCW()
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    print('Accuracy on test set: {}'.format(accuracy))

    attack = CarliniL2V2Container(
        mc,
        targeted=False,
        learning_rate=0.01,
        binary_search_steps=9,
        max_iter=1000,
        confidence=0.0,
        initial_const=0.01,
        batch_size=32,
        clip_values=(0.0, 1.0)
    )
    adv, y_adv, x_clean, y_clean = attack.generate(count=100)

    l2 = np.mean(get_l2_norm(adv, x_clean))
    print('L2 norm: {}'.format(l2))
    not_match = y_adv != y_clean
    success_rate = len(not_match[not_match == True]) / len(adv)
    print('Success rate: {}'.format(success_rate))

    accuracy = mc.evaluate(adv, y_clean)
    print('Accuracy on adv. examples: {}'.format(accuracy))


if __name__ == '__main__':
    main()
