import logging
import os

import numpy as np

from aad.basemodels import MnistCnnV2, ModelContainerPT
from aad.datasets import DATASET_LIST, DataContainer
from aad.utils import get_data_path, get_pt_model_filename, master_seed

logger = logging.getLogger(__name__)

SEED = 4096
BATCH_SIZE = 128
NAME = 'MNIST'
MAX_EPOCHS = 50


def main():
    master_seed(SEED)

    logger.info('Starting %s data container...', NAME)
    dc = DataContainer(DATASET_LIST[NAME], get_data_path())
    dc(shuffle=True)

    model = MnistCnnV2()
    filename = get_pt_model_filename(MnistCnnV2.__name__, NAME, MAX_EPOCHS)
    logger.debug('File name: %s', filename)

    mc = ModelContainerPT(model, dc)

    file_path = os.path.join('save', filename)
    if not os.path.exists(file_path):
        logger.debug('Expected initial loss: %f', np.log(10))
        mc.fit(epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
        mc.save(filename, overwrite=True)
    else:
        logger.info('Use saved parameters from %s', filename)
        mc.load(file_path)

    accuracy = mc.evaluate(dc.data_test_np, dc.label_test_np)
    logger.info('Accuracy on test set: %f', accuracy)


if __name__ == '__main__':
    main()
