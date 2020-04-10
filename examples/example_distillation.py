import logging
import os

from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import DistillationContainer
from aad.utils import get_data_path

logging.basicConfig(level=logging.DEBUG)

MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')

if __name__ == '__main__':
    Model = get_model('MnistCnnV2')
    model = Model()
    dc = DataContainer(DATASET_LIST['MNIST'], get_data_path())
    dc()
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    mc.fit(max_epochs=2)
    accuracy = mc.evaluate(dc.data_test_np, dc.label_test_np)
    print(f'Accuracy on test set: {accuracy}')

    distillation = DistillationContainer(mc, temperature=2.0)
    distillation.fit(max_epochs=100, batch_size=128)

    smooth_mc = distillation.get_def_model_container()
    accuracy = smooth_mc.evaluate(dc.data_test_np, dc.label_test_np)
    print(f'Accuracy on test set: {accuracy}')

    # -log(0.1) = 2.3025850929940455