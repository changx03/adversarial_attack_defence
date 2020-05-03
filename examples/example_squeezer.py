import logging
import os

import numpy as np

from aad.basemodels import MnistCnnV2, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import FeatureSqueezing
from aad.utils import get_data_path

logging.basicConfig(level=logging.DEBUG)


def build_model_filename(model_name, dataset, epochs):
    return f'{model_name}_{dataset}_e{epochs}.pt'


def build_squeezer_filename(model_name, data_name, max_epochs, filter_name):
    """
    Pre-train file example: MnistCnnV2_MNIST_e50_binary.pt
    """
    return f'{model_name}_{data_name}_e{max_epochs}_{filter_name}.pt'


def build_adv_filename(model_name, dataset, attack_name):
    return f'{model_name}_{dataset}_{attack_name}_adv.npy'


MODEL_NAME = 'MnistCnnV2'
DATASET = 'MNIST'
MAX_EPOCHS = 50
BIT_DEPTH = 8
SIGMA = 0.2
KERNEL_SIZE = 3

MODEL_FILE = os.path.join(
    'save',
    build_model_filename(MODEL_NAME, DATASET, MAX_EPOCHS)
)


def main():
    # load dataset and initial model
    model = MnistCnnV2()
    dc = DataContainer(DATASET_LIST[DATASET], get_data_path())
    dc(shuffle=True, normalize=True)
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on test set: {accuracy}')

    # train or load pretrained parameters
    squeezer = FeatureSqueezing(
        mc,
        ['median', 'normal', 'binary'],
        bit_depth=BIT_DEPTH,
        sigma=SIGMA,
        kernel_size=KERNEL_SIZE,
        pretrained=True
    )

    x_test = dc.x_test
    y_test = dc.y_test
    mc_binary = squeezer.get_def_model_container('binary')
    mc_median = squeezer.get_def_model_container('median')
    mc_normal = squeezer.get_def_model_container('normal')

    print('before fit')
    acc_bin = mc_binary.evaluate(
        squeezer.apply_binary_transform(x_test), y_test)
    print(f'Accuracy of binary squeezer: {acc_bin}')
    acc_med = mc_median.evaluate(
        squeezer.apply_median_transform(x_test), y_test)
    print(f'Accuracy of median squeezer: {acc_med}')
    acc_nor = mc_normal.evaluate(
        squeezer.apply_normal_transform(x_test), y_test)
    print(f'Accuracy of normal squeezer: {acc_nor}')

    if not squeezer.does_pretrained_exist(MODEL_FILE):
        squeezer.fit(max_epochs=MAX_EPOCHS, batch_size=128)

        print('after fit')
        acc_bin = mc_binary.evaluate(
            squeezer.apply_binary_transform(x_test), y_test)
        print(f'Accuracy of binary squeezer: {acc_bin}')
        acc_med = mc_median.evaluate(
            squeezer.apply_median_transform(x_test), y_test)
        print(f'Accuracy of median squeezer: {acc_med}')
        acc_nor = mc_normal.evaluate(
            squeezer.apply_normal_transform(x_test), y_test)
        print(f'Accuracy of normal squeezer: {acc_nor}')

        squeezer.save(MODEL_FILE, True)

    print('after load')
    squeezer.load(MODEL_FILE)
    acc_bin = mc_binary.evaluate(
        squeezer.apply_binary_transform(x_test), y_test)
    print(f'Accuracy of binary squeezer: {acc_bin}')
    acc_med = mc_median.evaluate(
        squeezer.apply_median_transform(x_test), y_test)
    print(f'Accuracy of median squeezer: {acc_med}')
    acc_nor = mc_normal.evaluate(
        squeezer.apply_normal_transform(x_test), y_test)
    print(f'Accuracy of normal squeezer: {acc_nor}')

    # load adversarial examples
    adv_list = ['FGSM', 'BIM', 'DeepFool', 'Carlini', 'Saliency']
    y_file = os.path.join(
        'save',
        f'{MODEL_NAME}_{DATASET}_{adv_list[0]}_y.npy')
    x_file = os.path.join(
        'save',
        f'{MODEL_NAME}_{DATASET}_{adv_list[0]}_x.npy')
    x = np.load(x_file, allow_pickle=False)
    y = np.load(y_file, allow_pickle=False)
    acc_og = mc.evaluate(x, y)
    acc_squeezer = squeezer.evaluate(x, y)
    print(f'Accuracy on clean set - OG: {acc_og}, Squeezer: {acc_squeezer}')

    for adv_name in adv_list:
        adv_file = os.path.join(
            'save',
            build_adv_filename(MODEL_NAME, DATASET, adv_name))
        adv = np.load(adv_file, allow_pickle=False)
        acc_og = mc.evaluate(adv, y)
        acc_squeezer = squeezer.evaluate(adv, y)
        print(
            f'Accuracy on {adv_name} set - OG: {acc_og}, Squeezer: {acc_squeezer}')


if __name__ == '__main__':
    main()
