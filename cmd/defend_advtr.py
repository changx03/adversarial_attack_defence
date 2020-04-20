import argparse as ap
import json
import logging
import os
import sys

import numpy as np

from aad.attacks import BIMContainer
from aad.basemodels import IrisNN, ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import AdversarialTraining, ApplicabilityDomainContainer
from aad.utils import get_data_path, get_time_str, master_seed
from cmd_utils import get_data_container, parse_model_filename, set_logging

LOG_NAME = 'DefAdvTr'
logger = logging.getLogger(LOG_NAME)


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='a file which contains a pretrained model. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-e', '--epoch', type=int, required=True,
        help='the number of max epochs for training')
    parser.add_argument(
        '-r', '--ratio', type=float, required=True,
        help='the percentage of adversarial examples mix to the training set.')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=128, help='batch size')
    parser.add_argument(
        '-s', '--seed', type=int, default=4096,
        help='the seed for random number generator')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='set logger level to debug')
    parser.add_argument(
        '-l', '--savelog', action='store_true', default=False,
        help='save logging file')
    parser.add_argument(
        '-B', '--bim', action='store_true', default=False,
        help='Apply BIM attack')
    parser.add_argument(
        '-C', '--carlini', action='store_true', default=False,
        help='Apply Carlini L2 attack')
    parser.add_argument(
        '-D', '--deepfool', action='store_true', default=False,
        help='Apply DeepFool attack')
    parser.add_argument(
        '-F', '--fgsm', action='store_true', default=False,
        help='Apply FGSM attack')
    parser.add_argument(
        '-S', '--saliency', action='store_true', default=False,
        help='Apply Saliency Map attack')
    args = parser.parse_args()
    model_file = args.model
    max_epochs = args.epoch
    ratio = args.ratio
    batch_size = args.batchsize
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog

    model_name, data_name = parse_model_filename(model_file)

    # Which attack should apply?
    attack_list = []
    if args.bim:
        attack_list.append('BIM')
    if args.carlini:
        attack_list.append('Carlini')
    if args.deepfool:
        attack_list.append('DeepFool')
    if args.fgsm:
        attack_list.append('FGSM')
    if args.saliency:
        attack_list.append('Saliency')

    y_file = os.path.join(
        'save', f'{model_name}_{data_name}_{attack_list[0]}_y.npy')
    attack_files = [
        os.path.join(
            'save', f'{model_name}_{data_name}_{attack_list[0]}_x.npy')
    ]
    for attack_name in attack_list:
        attack_files.append(os.path.join(
            'save', f'{model_name}_{data_name}_{attack_name}_adv.npy'))
    # the 1st file this the clean inputs
    attack_list = ['clean'] + attack_list

    # Do I need train the discriminator?
    need_train = False
    pretrain_file = f'AdvTrain_{model_name}_{data_name}.pt'
    if not os.path.exists(os.path.join('save', pretrain_file)):
        need_train = True

    # set logging config. Run this before logging anything!
    set_logging(LOG_NAME, data_name, verbose, save_log)

    # show parameters
    print('[DefAdvTr] Running adversarial training on {}...'.format(model_name))
    logger.info('Start at    : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file  :%s', model_file)
    logger.info('model       :%s', model_name)
    logger.info('dataset     :%s', data_name)
    logger.info('max_epochs  :%d', max_epochs)
    logger.info('ratio  :%d', ratio)
    logger.info('batch_size  :%d', batch_size)
    logger.info('seed        :%d', seed)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('need train  :%r', need_train)
    logger.info('attacks     :%s', ', '.join(attack_list))

    # check files
    for file_name in [model_file, y_file] + attack_files:
        if not os.path.exists(file_name):
            logger.error('%s does not exist!', file_name)
            raise FileNotFoundError('{} does not exist!'.format(file_name))

    # reset seed
    master_seed(seed)

    # select data
    dc = get_data_container(
        data_name,
        use_shuffle=True,
        use_normalize=True,
    )

    # select a model
    Model = get_model(model_name)
    model = Model()
    if data_name in ('BankNote', 'HTRU2', 'Iris', 'WheatSeed'):
        num_classes = dc.num_classes
        num_features = dc.dim_data[0]
        model = IrisNN(
            num_features=num_features,
            hidden_nodes=num_features*4,
            num_classes=num_classes)
    classifier_mc = ModelContainerPT(model, dc)
    classifier_mc.load(model_file)
    accuracy = classifier_mc.evaluate(dc.x_test, dc.y_test)
    logger.info('Accuracy on test set: %f', accuracy)

    attack = BIMContainer(
        classifier_mc,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False)

    adv_trainer = AdversarialTraining(classifier_mc, [attack])
    if need_train:
        adv_trainer.fit(max_epochs=max_epochs, batch_size=batch_size, ratio=ratio)
        adv_trainer.save(pretrain_file, overwrite=True)
    else:
        adv_trainer.load(os.path.join('save', pretrain_file))

    y = np.load(y_file, allow_pickle=False)
    for i in range(len(attack_list)):
        adv_file = attack_files[i]
        adv_name = attack_list[i]
        logger.debug('Load %s...', adv_file)
        adv = np.load(adv_file, allow_pickle=False)
        accuracy = classifier_mc.evaluate(adv, y)
        logger.info('Accuracy on %s set: %f', adv_name, accuracy)
        blocked_indices = adv_trainer.detect(adv, return_passed_x=False)
        logger.info('Blocked %d/%d samples on %s',
                    len(blocked_indices), len(adv), adv_name)


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/defend_advtr.py -vl -e 200 -r 0.5 -m ./save/IrisNN_Iris_e200.pt -BCDF
    $ python ./cmd/defend_advtr.py -vl -e 200 -r 0.25 -m ./save/BCNN_BreastCancerWisconsin_e200.pt  -BCDF
    $ python ./cmd/defend_advtr.py -vl -e 200 -r 0.25 -m ./save/IrisNN_BankNote_e200.pt  -BCDF
    $ python ./cmd/defend_advtr.py -vl -e 200 -r 0.25 -m ./save/IrisNN_HTRU2_e200.pt  -BCDF
    $ python ./cmd/defend_advtr.py -vl -e 200 -r 0.5 -m ./save/IrisNN_WheatSeed_e300.pt  -BCDF
    $ python ./cmd/defend_advtr.py -vl -e 30 -r 0.25 -m ./save/MnistCnnV2_MNIST_e50.pt  -BCDFS
    $ python ./cmd/defend_advtr.py -vl -e 30 -r 0.25 -m ./save/CifarCnn_CIFAR10_e50.pt  -BCDFS
    $ python ./cmd/defend_advtr.py -vl -e 30 -r 0.25 -m ./save/CifarResnet50_CIFAR10_e30.pt  -BCDFS
    $ python ./cmd/defend_advtr.py -vl -e 30 -r 0.25 -m ./save/CifarResnet50_SVHN_e30.pt  -BCDFS
    """
    main()
    print('[DefAdvTr] Task completed!')
