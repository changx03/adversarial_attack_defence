import logging
import os

import numpy as np

from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import DistillationContainer
from aad.utils import get_data_path

logging.basicConfig(level=logging.DEBUG)


def build_model_filename(model_name, dataset, epochs):
    return f'{model_name}_{dataset}_e{epochs}.pt'


def build_distill_filename(model_name, dataset, epochs, temp):
    return f'distill_{model_name}_{dataset}_t{temp}_e{epochs}.pt'


def build_adv_filename(model_name, dataset, attack_name):
    return f'{model_name}_{dataset}_{attack_name}_adv.npy'


MODEL_NAME = 'MnistCnnV2'
DATASET = 'MNIST'
MAX_EPOCHS = 50
# 20 is the recommended value from original paper
TEMPERATURE = 20

MODEL_FILE = os.path.join(
    'save',
    build_model_filename(MODEL_NAME, DATASET, MAX_EPOCHS)
)
DISTILL_FILE = build_distill_filename(
    MODEL_NAME, DATASET, MAX_EPOCHS, TEMPERATURE)


def main():
    # load dataset and initial model
    Model = get_model(MODEL_NAME)
    model = Model()
    dc = DataContainer(DATASET_LIST[DATASET], get_data_path())
    dc()
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on test set: {accuracy}')

    # train or load distillation model
    distillation = DistillationContainer(
        mc, Model(), temperature=TEMPERATURE, pretrained=False)

    distill_path = os.path.join('save', DISTILL_FILE)
    if not os.path.exists(distill_path):
        distillation.fit(max_epochs=MAX_EPOCHS, batch_size=128)
        distillation.save(DISTILL_FILE, True)
    else:
        distillation.load(distill_path)

    smooth_mc = distillation.get_def_model_container()
    accuracy = smooth_mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on test set: {accuracy}')

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
    acc_distill = smooth_mc.evaluate(x, y)
    print(f'Accuracy on clean set - OG: {acc_og}, Distill: {acc_distill}')

    for adv_name in adv_list:
        adv_file = os.path.join(
            'save',
            build_adv_filename(MODEL_NAME, DATASET, adv_name))
        adv = np.load(adv_file, allow_pickle=False)
        acc_og = mc.evaluate(adv, y)
        acc_distill = smooth_mc.evaluate(adv, y)
        print(f'Accuracy on {adv_name} set - OG: {acc_og}, Distill: {acc_distill}')


if __name__ == '__main__':
    main()
