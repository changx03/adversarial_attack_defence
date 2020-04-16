import os
import logging

import numpy as np

from aad.attacks import CarliniL2V2Container
from aad.basemodels import ModelContainerPT, IrisNN
from aad.utils import master_seed, get_data_path, get_l2_norm, get_range
from aad.datasets import DATASET_LIST, DataContainer

logging.basicConfig(level=logging.DEBUG)

DATA_NAME = 'WheatSeed'
SEED = 4096
MODEL_FILE = os.path.join('save', 'IrisNN_WheatSeed_e300.pt')


def main():
    master_seed(SEED)

    dataset = DATASET_LIST[DATA_NAME]
    dc = DataContainer(dataset, get_data_path())
    dc(shuffle=True, normalize=True)
    print('# of trainset: {}, # of testset: {}'.format(
        len(dc.x_train), 
        len(dc.x_test)))
    num_classes = dc.num_classes
    num_features = dc.dim_data[0]
    model = IrisNN(
        num_features=num_features,
        hidden_nodes=num_features*4,
        num_classes=num_classes)
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    print('Accuracy on test set: {}'.format(accuracy))

    clip_values = get_range(dc.x_train, is_image=False)
    print('clip_values', clip_values)

    attack = CarliniL2V2Container(
        mc,
        targeted=False,
        learning_rate=0.01,
        binary_search_steps=9,
        max_iter=1000,
        confidence=0.0,
        initial_const=0.01,
        c_range=(0, 1e4),
        batch_size=16,
        clip_values=clip_values
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
