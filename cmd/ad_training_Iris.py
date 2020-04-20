import logging
import os

import numpy as np

from aad.attacks import BIMContainer
from aad.basemodels import ModelContainerPT, IrisNN
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import AdversarialTraining
from aad.utils import get_data_path
from cmd_utils import set_logging

logger = logging.getLogger('AdvTrain')


def main():
    data_name = 'Iris'
    set_logging('advTraining', data_name, True, True)

    dc = DataContainer(DATASET_LIST[data_name], get_data_path())
    dc()

    model_file = os.path.join('save', 'IrisNN_Iris_e200.pt')
    num_features = dc.dim_data[0]
    num_classes = dc.num_classes
    classifier = IrisNN(
        num_features=num_features,
        hidden_nodes=num_features*4,
        num_classes=num_classes,
    )

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
    # adv_trainer.fit(max_epochs=100, batch_size=64, ratio=1)
    # adv_trainer.save('AdvTrain_IrisNN_Iris', overwrite=True)

    file_name = os.path.join('save', 'AdvTrain_IrisNN_Iris.pt')
    adv_trainer.load(file_name)

    x = np.load(os.path.join('save', 'IrisNN_Iris_BIM_x.npy'),
                allow_pickle=False)
    y = np.load(os.path.join('save', 'IrisNN_Iris_BIM_y.npy'),
                allow_pickle=False)
    blocked_indices = adv_trainer.detect(x, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(x), 'clean')

    adv = np.load(os.path.join(
        'save', 'IrisNN_Iris_BIM_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'BIM', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'BIM')

    adv = np.load(os.path.join(
        'save', 'IrisNN_Iris_Carlini_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'Carlini', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'Carlini')

    adv = np.load(os.path.join(
        'save', 'IrisNN_Iris_DeepFool_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'DeepFool', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'DeepFool')

    adv = np.load(os.path.join(
        'save', 'IrisNN_Iris_FGSM_adv.npy'), allow_pickle=False)
    accuracy = classifier_mc.evaluate(adv, y)
    logger.info('Accuracy on %s set: %f', 'FGSM', accuracy)
    blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(adv), 'FGSM')


if __name__ == '__main__':
    main()
