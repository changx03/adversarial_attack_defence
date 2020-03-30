"""
This file provides the script of terminal interface to train and save the model
"""
import argparse as ap
import logging
import os

from aad.basemodels import (BCNN, CifarCnn, IrisNN, MnistCnn_v2,
                            ModelContainerPT, get_model)
from aad.datasets import DATASET_LIST, DataContainer, get_dataset_list
from aad.utils import (get_data_path, get_pt_model_filename, get_time_str,
                       master_seed)

logger = logging.getLogger('train')

AVALIABLE_MODELS = (
    'BCNN',
    'CifarCnn',
    'IrisNN',
    'MnistCnnCW',
    'MnistCnn_v2',
)


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, choices=get_dataset_list(),
        help='the dataset you want to train')
    parser.add_argument(
        '-o', '--ofile', type=str,
        help='the filename will be used to store model parameters')
    parser.add_argument(
        '-e', '--epoch', type=int, default=5,
        help='the number of max epochs for training')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=128, help='batch size')
    parser.add_argument(
        '-s', '--seed', type=int, default=4096,
        help='the seed for random number generator')
    parser.add_argument(
        '-H', '--shuffle', type=bool, default=True, help='shuffle the dataset')
    parser.add_argument(
        '-n', '--normalize', type=bool, default=True,
        help='apply zero mean and scaling to the dataset (for numeral dataset only)')
    parser.add_argument(
        '-m', '--model', type=str, choices=AVALIABLE_MODELS,
        help='select a model to train the data')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='set logger level to debug')
    parser.add_argument(
        '-l', '--savelog', action='store_true', default=False,
        help='save logging file')
    parser.add_argument(
        '-O', '--overwrite', action='store_true', default=False,
        help='overwrite the existing file')
    args = parser.parse_args()
    dname = args.dataset
    filename = args.ofile
    max_epochs = args.epoch
    batch_size = args.batchsize
    seed = args.seed
    use_shuffle = args.shuffle
    use_normalize = args.normalize
    model_name = args.model
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # set logging config. Run this before logging anything!
    log_lvl = logging.DEBUG if verbose else logging.INFO
    time_str = get_time_str()
    if save_log:
        log_filename = f'train_{dname}_{time_str}.log'
        if not os.path.exists('log'):
            os.makedirs('log')
        logging.basicConfig(
            filename=os.path.join('log', log_filename),
            format='%(asctime)s:%(levelname)s:%(module)s:%(message)s',
            level=log_lvl)
    else:
        logging.basicConfig(level=log_lvl)

    # show parameters
    logger.info('Start at: %s', time_str)
    logger.info('RECEIVED PARAMETERS:')
    logger.info('dataset       :%s', dname)
    logger.info('filename      :%s', filename)
    logger.info('max_epochs    :%d', max_epochs)
    logger.info('batch_size    :%d', batch_size)
    logger.info('seed          :%d', seed)
    logger.info('use_shuffle   :%r', use_shuffle)
    logger.info('use_normalize :%r', use_normalize)
    logger.info('model_name    :%s', model_name)
    logger.info('verbose       :%r', verbose)
    logger.info('save_log      :%r', save_log)
    logger.info('overwrite     :%r', overwrite)

    # set DataContainer
    dataset = DATASET_LIST[dname]
    dc = DataContainer(dataset, get_data_path())
    if dname in ('MNIST', 'CIFAR10', 'SVHN'):
        dc(shuffle=use_shuffle)
    elif dname == 'Iris':
        dc(shuffle=use_shuffle, normalize=use_normalize, size_train=0.6)
    elif dname in ('BankNote', 'BreastCancerWisconsin', 'HTRU2', 'WheatSeed'):
        dc(shuffle=use_shuffle, normalize=use_normalize)
    else:
        raise AttributeError('Received unknown dataset "{}"'.format(dname))

    # select a model
    model = None
    if model_name is not None:
        Model = get_model(model_name)
        model = Model()
    else:
        if dname == 'MNIST':
            model = MnistCnn_v2()
        elif dname == 'CIFAR10':
            model = CifarCnn()
        elif dname == 'BreastCancerWisconsin':
            model = BCNN()
        elif dname in ('BankNote', 'HTRU2', 'Iris', 'WheatSeed'):
            num_classes = dataset['num_classes']
            num_features = dataset['dim_data'][0]
            model = IrisNN(
                num_features=num_features,
                hidden_nodes=num_features*4,
                num_classes=num_classes)

    if model is None:
        raise AttributeError('Cannot find model!')
    modelname = model.__class__.__name__
    logger.info('Select %s model', modelname)

    # set ModelContainer and train the model
    master_seed(seed)
    mc = ModelContainerPT(model, dc)
    mc.fit(epochs=max_epochs, batch_size=batch_size)

    # save
    if not os.path.exists('save'):
        os.makedirs('save')
    if filename is None:
        filename = get_pt_model_filename(modelname, dname, max_epochs)
    logger.debug('File name: %s', filename)
    mc.save(filename, overwrite=overwrite)

    # test result
    file_path = os.path.join('save', filename)
    logger.debug('Use saved parameters from %s', filename)
    mc.load(file_path)
    accuracy = mc.evaluate(dc.data_test_np, dc.label_test_np)
    logger.info('Accuracy on test set: %f', accuracy)


if __name__ == "__main__":
    """
    Examples:
    $ python ./cmd/train.py -d MNIST -e 50 -vO
    $ python ./cmd/train.py -d CIFAR10 -e 50 -vO
    $ python ./cmd/train.py -d BreastCancerWisconsin -e 200 -vO
    $ python ./cmd/train.py -d BankNote -e 200 -vO
    $ python ./cmd/train.py -d HTRU2 -e 200 -vO
    $ python ./cmd/train.py -d Iris -e 200 -vO
    $ python ./cmd/train.py -d WheatSeed -e 200 -vO
    """
    main()
