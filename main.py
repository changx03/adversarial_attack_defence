import sys

from attacks import AttackContainer
from basemodels import ModelContainer
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

    DATA_ROOT = 'data'
    BATCH_SIZE = 64
    TYPE = 'image'  # image or quantitative
    # image: 'MNIST', 'CIFAR10', 'SVHN'
    # quantitative: 'BankNote', 'BreastCancerWisconsin', 'WheatSeed', 'HTRU2'
    NAME = 'SVHN'

    print(f'Start {NAME} data container')
    IMAGE_DATASET = DATASET_LIST[TYPE][NAME]
    image_data = DataContainer(IMAGE_DATASET, DATA_ROOT)
    image_data(batch_size=BATCH_SIZE)
    print(len(image_data))
    print('train', image_data.dim_train)
    print(image_data.label_test_np[:16])
    print()

    # model = ModelContainer()
    # model()
    # attack = AttackContainer()
    # attack()
    # defence = DefenceContainer()
    # defence()


if __name__ == '__main__':
    # make sure the program runs in the correct environment
    print(sys.version)
    print(*sys.path, sep='\n')
    print()

    main()
