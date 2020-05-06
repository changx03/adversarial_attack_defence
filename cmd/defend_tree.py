import argparse as ap
import json
import logging
import os
import sys

import numpy as np
from art.attacks import DecisionTreeAttack
from art.classifiers import SklearnClassifier
from sklearn.tree import ExtraTreeClassifier

from aad.basemodels import ModelContainerTree
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_time_str, master_seed
from cmd_utils import get_data_container, set_logging

LOG_NAME = 'DefTree'
logger = logging.getLogger(LOG_NAME)


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help='Name of the dataset')
    parser.add_argument(
        '-p', '--param', type=str, required=True,
        help='a JSON config file which contains the parameters for the attacks')
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
        '-F', '--fgsm', action='store_true', default=False,
        help='Apply FGSM attack')
    parser.add_argument(
        '-B', '--bim', action='store_true', default=False,
        help='Apply BIM attack')
    parser.add_argument(
        '-D', '--deepfool', action='store_true', default=False,
        help='Apply DeepFool attack')
    parser.add_argument(
        '-C', '--carlini', action='store_true', default=False,
        help='Apply Carlini L2 attack')
    args = parser.parse_args()
    data_name = args.dataset
    param_file = args.param
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog

    # set logging config. Run this before logging anything!
    set_logging(LOG_NAME, data_name, verbose, save_log)

    # Which attack should apply?
    attack_list = []
    if args.fgsm:
        attack_list.append('FGSM')
    if args.bim:
        attack_list.append('BIM')
    if args.deepfool:
        attack_list.append('DeepFool')
    if args.carlini:
        attack_list.append('Carlini')

    # Quit, if there is nothing to do.
    if len(attack_list) == 0:
        logger.warning('Neither received any filter nor any attack. Exit')
        sys.exit(0)

    if data_name in ('BankNote', 'HTRU2', 'Iris', 'WheatSeed'):
        model_name = 'IrisNN'
    if data_name == 'BreastCancerWisconsin':
        model_name = 'BCNN'

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

    # load parameters for Applicability Domain
    with open(param_file) as param_json:
        params = json.load(param_json)

    # show parameters
    print(f'[{LOG_NAME}] Running feature squeezing on {model_name}...')
    logger.info('Start at    : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model       :%s', model_name)
    logger.info('dataset     :%s', data_name)
    logger.info('param file  :%s', param_file)
    logger.info('seed        :%d', seed)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('attacks     :%s', ', '.join(attack_list))
    logger.debug('params     : %s', str(params))

    # check files
    for file_name in [y_file] + attack_files:
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

    # train the model
    classifier = ExtraTreeClassifier(
        criterion='gini',
        splitter='random',
    )
    mc = ModelContainerTree(classifier, dc)
    mc.fit()

    x = np.load(attack_files[0], allow_pickle=False)
    art_classifier = SklearnClassifier(classifier)
    attack = DecisionTreeAttack(art_classifier)
    adv = attack.generate(x)

    ad = ApplicabilityDomainContainer(
        mc, mc.hidden_model, **params)
    ad.fit()

    # generate adversarial examples
    y = np.load(y_file, allow_pickle=False)

    accuracy = mc.evaluate(adv, y)
    logger.info('Accuracy on DecisionTreeAttack set: %f', accuracy)
    blocked_indices = ad.detect(adv)
    logger.info('Blocked %d/%d samples on DecisionTreeAttack',
                len(blocked_indices), len(adv))

    # traverse other attacks
    for i in range(len(attack_list)):
        adv_file = attack_files[i]
        adv_name = attack_list[i]
        logger.debug('Load %s...', adv_file)
        adv = np.load(adv_file, allow_pickle=False)
        accuracy = mc.evaluate(adv, y)
        logger.info('Accuracy on %s set: %f', adv_name, accuracy)
        blocked_indices = ad.detect(adv, return_passed_x=False)
        logger.info('Blocked %d/%d samples on %s',
                    len(blocked_indices), len(adv), adv_name)


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/defend_tree.py -v -p ./cmd/AdParamsNumeral.json -d Iris -BCDF
    $ python ./cmd/defend_tree.py -v -p ./cmd/AdParamsNumeral.json -d BreastCancerWisconsin -BCDF
    $ python ./cmd/defend_tree.py -v -p ./cmd/AdParamsNumeral.json -d BankNote -BCDF
    $ python ./cmd/defend_tree.py -v -p ./cmd/AdParamsNumeral.json -d HTRU2 -BCDF
    $ python ./cmd/defend_tree.py -v -p ./cmd/AdParamsNumeral.json -d WheatSeed -BCDF
    """
    main()
    print(f'[{LOG_NAME}] Task completed!')
