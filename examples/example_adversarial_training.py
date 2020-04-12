import logging
import os

from aad.attacks import BIMContainer
from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import AdversarialTraining
from aad.utils import get_data_path

logging.basicConfig(level=logging.DEBUG)

MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')


def main():
    Model = get_model('MnistCnnV2')
    classifier = Model()

    dc = DataContainer(DATASET_LIST['MNIST'], get_data_path())
    dc()
    classifier_mc = ModelContainerPT(classifier, dc)
    classifier_mc.load(MODEL_FILE)
    accuracy = classifier_mc.evaluate(dc.data_test_np, dc.label_test_np)
    print(f'Accuracy on test set: {accuracy}')

    attack = BIMContainer(
        classifier_mc,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False)

    adv_trainer = AdversarialTraining(classifier_mc, [attack])
    adv_trainer.fit(max_epochs=5, batch_size=128, ratio=0.1)
    discriminator = adv_trainer.get_def_model_container()
    print(discriminator.accuracy_test)


if __name__ == '__main__':
    main()
