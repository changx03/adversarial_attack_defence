import sys

from attacks import AttackContainer
from basemodels import TorchModelContainer, MnistCnnCW
from datasets import (DATASET_LIST, DataContainer, get_image_list,
                      get_quantitative_list)
from defences import DefenceContainer


def main():
    print('Avaliable image datasets:')
    print(get_image_list())
    print()
    print('Avaliable quantitative datasets:')
    print(get_quantitative_list())
    print()

    # 1. choose a dataset
    DATA_ROOT = 'data'
    BATCH_SIZE = 64
    TYPE = 'image'  # image or quantitative
    # image: 'MNIST', 'CIFAR10', 'SVHN'
    # quantitative: 'BankNote', 'BreastCancerWisconsin', 'WheatSeed', 'HTRU2'
    NAME = 'MNIST'

    print(f'Starting {NAME} data container...')
    IMAGE_DATASET = DATASET_LIST[TYPE][NAME]
    dc = DataContainer(IMAGE_DATASET, DATA_ROOT)
    dc(BATCH_SIZE)

    # 2. choose a model
    model = MnistCnnCW()

    # 3. train/load the model
    # train, save, load
    mc = TorchModelContainer(model, dc)
    mc.fit(epochs=5)
    
    # 4. populate adversarial examples
    # train, save, load
    # attack = AttackContainer()
    # attack()


    # 5. test defence method
    # train, save, load
    # defence = DefenceContainer()
    # defence()


if __name__ == '__main__':
    # make sure the program runs in the correct environment
    print(sys.version)
    print(*sys.path, sep='\n')
    print()

    main()
