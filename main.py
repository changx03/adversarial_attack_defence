from datasets import DataContainer, DATASET_LIST
from basemodels import ModelContainer
from attacks import AttackContainer
from defences import DefenceContainer
import sys


def main():
    IMAGE_DATASET = DATASET_LIST['image']['MNIST']
    DATA_ROOT = 'data'
    BATCH_SIZE = 64

    image_data = DataContainer(IMAGE_DATASET, DATA_ROOT)
    image_data(batch_size=BATCH_SIZE)
    print(len(image_data))
    print(image_data.dim_train)
    print(image_data.label_test_np[:16])

    NUM_DATASET = DATASET_LIST['quantitative']['BankNote']
    num_data = DataContainer(NUM_DATASET, DATA_ROOT)
    num_data(BATCH_SIZE, normalize=True)

    model = ModelContainer()
    model()
    attack = AttackContainer()
    attack()
    defence = DefenceContainer()
    defence()


if __name__ == '__main__':
    # make sure the program runs in the correct environment
    print(sys.version)
    print(*sys.path, sep='\n')

    main()
