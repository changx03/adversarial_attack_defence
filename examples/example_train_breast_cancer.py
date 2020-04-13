import logging
import os

import numpy as np

from aad.basemodels import BCNN, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, get_pt_model_filename, master_seed

logger = logging.getLogger(__name__)

SEED = 4096
BATCH_SIZE = 256
NAME = 'BreastCancerWisconsin'
MAX_EPOCHS = 200


def main():
    master_seed(SEED)

    logger.info('Starting %s data container...', NAME)
    dc = DataContainer(DATASET_LIST[NAME], get_data_path())
    dc(shuffle=True, normalize=True)

    num_features = dc.dim_data[0]
    num_classes = dc.num_classes
    print('Features:', num_features)
    print('Classes:', num_classes)
    model = BCNN(num_features, num_classes)
    filename = get_pt_model_filename(BCNN.__name__, NAME, MAX_EPOCHS)
    logger.debug('File name: %s', filename)

    mc = ModelContainerPT(model, dc)

    file_path = os.path.join('save', filename)
    if not os.path.exists(file_path):
        logger.debug('Expected initial loss: %f', np.log(dc.num_classes))
        mc.fit(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
        mc.save(filename, overwrite=True)
    else:
        logger.info('Use saved parameters from %s', filename)
        mc.load(file_path)

    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    logger.info('Accuracy on test set: %f', accuracy)


if __name__ == '__main__':
    main()
