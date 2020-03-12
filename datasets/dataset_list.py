DATASET_LIST = {
    'image': {
        'MNIST': {
            'name': 'MNIST',
            'type': 'image',
            'size': 7e4,
            'num_classes': 10,
            'dim_data': (1, 28, 28),
        },
        'CIFAR10': {
            'name': 'CIFAR10',
            'type': 'image',
            'size': 6e4,
            'num_classes': 10,
            'dim_data': (3, 32, 32),
        },
        'SVHM': {
            'name': 'SVHM',
            'type': 'image',
            'size': 73257 + 26032,
            'num_classes': 10,
            'dim_data': (3, 32, 32),
        },
    },
    'quantitative': {
        'BankNote': {
            'name': 'BankNote',
            'type': 'quantitative',
            'size': 1372,
            'num_classes': 2,
            'dim_data': (4,),
        },
        'BreastCancerWisconsin': {
            'name': 'BreastCancerWisconsin',
            'type': 'quantitative',
            'size': 569,
            'num_classes': 2,
            'dim_data': (30,),
        },
        'WheatSeed': {
            'name': 'WheatSeed',
            'type': 'quantitative',
            'size': 210,
            'num_classes': 3,
            'dim_data': (7,),
        },
        'HTRU2': {
            'name': 'HTRU2',
            'type': 'quantitative',
            'size': 17898,
            'num_classes': 2,
            'dim_data': (8,),
        },
    },
}

def get_image_list():
    return list(DATASET_LIST['image'].keys())

def get_quantitative_list():
    return list(DATASET_LIST['quantitative'].keys())