import logging
import os

import numpy as np

from aad.attacks import BIMContainer
from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import AdversarialTraining
from aad.utils import get_data_path
from cmd_utils import set_logging

logger = logging.getLogger('AdvTrain')


def main():
    data_name = 'MNIST'
    set_logging('advTraining', data_name, True, True)

    model_file = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')
    Model = get_model('MnistCnnV2')
    classifier = Model()

    dc = DataContainer(DATASET_LIST[data_name], get_data_path())
    dc()
    classifier_mc = ModelContainerPT(classifier, dc)
    classifier_mc.load(model_file)
    accuracy = classifier_mc.evaluate(dc.x_test, dc.y_test)
    logger.info('Accuracy on test set: %f', accuracy)

    attack = BIMContainer(
        classifier_mc,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False)

    adv_trainer = AdversarialTraining(classifier_mc, [attack])
    # adv_trainer.fit(max_epochs=30, batch_size=128, ratio=0.1)
    # adv_trainer.save('AdvTrain_MnistCnnV2_MNIST', overwrite=True)

    file_name = os.path.join('save', 'AdvTrain_MnistCnnV2_MNIST.pt')
    adv_trainer.load(file_name)

    x = np.load(os.path.join('save', 'MnistCnnV2_MNIST_BIM_x.npy'),
                allow_pickle=False)
    y = np.load(os.path.join('save', 'MnistCnnV2_MNIST_BIM_y.npy'),
                allow_pickle=False)
    blocked_indices = adv_trainer.detect(x, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(x), 'clean')

    adv = np.load(os.path.join(
        'save', 'MnistCnnV2_MNIST_BIM_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'BIM', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'BIM')

    adv = np.load(os.path.join(
        'save', 'MnistCnnV2_MNIST_Carlini_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'Carlini', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'Carlini')

    adv = np.load(os.path.join(
        'save', 'MnistCnnV2_MNIST_DeepFool_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'DeepFool', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'DeepFool')

    adv = np.load(os.path.join(
        'save', 'MnistCnnV2_MNIST_FGSM_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'FGSM', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'FGSM')

    adv = np.load(os.path.join(
        'save', 'MnistCnnV2_MNIST_Saliency_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'Saliency', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'Saliency')


if __name__ == '__main__':
    main()
