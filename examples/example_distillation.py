import logging
import os

from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import DistillationContainer
from aad.utils import get_data_path

logging.basicConfig(level=logging.DEBUG)

MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')


def main():
    Model = get_model('MnistCnnV2')
    model = Model()
    dc = DataContainer(DATASET_LIST['MNIST'], get_data_path())
    dc()
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    # mc.fit(max_epochs=2)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on test set: {accuracy}')

    distillation = DistillationContainer(
        mc, Model(), temperature=1.0, pretrained=False)

    print('Expected initial loss = -log(1/num_classes) = 2.3025850929940455')
    distillation.fit(max_epochs=8, batch_size=128)

    smooth_mc = distillation.get_def_model_container()
    accuracy = smooth_mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on test set: {accuracy}')


if __name__ == '__main__':
    main()
