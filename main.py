from datasets import DataContainer, DATASET_LIST
from basemodels import ModelContainer
from attacks import AttackContainer
from defences import DefenceContainer


def main():
    MY_DATASET = DATASET_LIST['image']['MNIST']
    DATA_ROOT = 'data'
    BATCH_SIZE = 64

    data = DataContainer(MY_DATASET, DATA_ROOT)
    data(batch_size=BATCH_SIZE)
    print(len(data))
    print(data.train_dim)
    print(data.label_test_np[:16])

    model = ModelContainer()
    model()

    attack = AttackContainer()
    attack()

    defence = DefenceContainer()
    defence()


if __name__ == '__main__':
    main()
