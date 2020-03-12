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
    TYPE = 'quantitative'  # image or quantitative
    # image: 'MNIST', 'CIFAR10', 'SVHN'
    # quantitative: 'BankNote', 'BreastCancerWisconsin', 'WheatSeed', 'HTRU2'
    NAME = 'BreastCancerWisconsin'

    print(f'Start {NAME} data container')
    IMAGE_DATASET = DATASET_LIST[TYPE][NAME]
    data_container = DataContainer(IMAGE_DATASET, DATA_ROOT)
    data_container(batch_size=BATCH_SIZE, normalize=False)
    print(len(data_container))
    print('train', data_container.dim_train)
    print(data_container.label_test_np[:16])

    print('Compute mean and std from train set')
    data_np = data_container.data_train_np
    print('mean: [', end='')
    for m in data_np.mean(axis=0):
        print(f'{m:.4f}', end=', ')
    print('],')
    print('std: [', end='')
    for s in data_np.std(axis=0):
        print(f'{s:.4f}', end=', ')
    print('],')

    print(data_container.mean)
    print(data_container.std)

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
