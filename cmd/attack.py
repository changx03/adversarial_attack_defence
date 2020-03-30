import argparse as ap
import logging
import os
import sys

import numpy as np

import aad.attacks as attacks
from aad.basemodels import get_model
from aad.utils import get_time_str
from cmd_utils import get_data_container, set_logging

logger = logging.getLogger('attack')


def parse_model_filename(filename):
    """
    Parses the filename of a trained model. The filename should in
    "<model>_<dataset>_e<max epochs>[_<date>].pt" format'.
    """
    dirname = os.path.split(filename)
    arr = dirname[-1].split('_')
    model_name = arr[0]
    dataset_name = arr[1]
    return model_name, dataset_name


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='pretrained model file name. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-n', '--number', type=int, default=100,
        help='the number of adversarial examples want to generate. (if more than test set, it uses all test examples.)')
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
        '-O', '--overwrite', action='store_true', default=False,
        help='overwrite the existing file')
    parser.add_argument(
        '-F', '--fgsm', action='store_true', default=False,
        help='Apply FGSM attack')
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
        '-S', '--saliency', action='store_true', default=False,
        help='Apply Saliency Map attack')
    args = parser.parse_args()
    model_filename = args.model
    num_adv = args.number
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # Which attack should apply?
    # use binary encoding for attacks
    my_attacks = np.zeros(5, dtype=np.int8)
    attack_list = np.array(
        ['FGSM', 'BIM', 'Carlini L2', 'DeepFool', 'Saliency'])
    my_attacks[0] = 1 if args.fgsm else 0
    my_attacks[1] = 1 if args.bim else 0
    my_attacks[2] = 1 if args.carlini else 0
    my_attacks[3] = 1 if args.deepfool else 0
    my_attacks[4] = 1 if args.saliency else 0
    selected_attacks = attack_list[np.where(my_attacks == 1)[0]]

    # check file
    if not os.path.exists(model_filename):
        raise FileNotFoundError('{} does not exist!'.format(model_filename))
    dirname = os.path.dirname(model_filename)

    model_name, dname = parse_model_filename(model_filename)

    # set logging config. Run this before logging anything!
    set_logging(dname, verbose, save_log)

    # show parameters
    logger.info('Start at: %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file :%s', model_filename)
    logger.info('model      :%s', model_name)
    logger.info('dataset    :%s', dname)
    logger.info('num_adv    :%r', num_adv)
    logger.info('seed       :%d', seed)
    logger.info('verbose    :%r', verbose)
    logger.info('save_log   :%r', save_log)
    logger.info('overwrite  :%r', overwrite)
    logger.info('dirname    :%r', dirname)
    logger.info('attacks    : %s', ', '.join(selected_attacks))

    if len(selected_attacks) == 0:
        logger.warning('No attack is selected. Exit.')
        sys.exit(0)

    # set DataContainer
    dc = get_data_container(dname)

    Model = get_model(model_name)
    model = Model()
    logger.info('Select %s model', model.__class__.__name__)


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/attack.py -m ./save/IrisNN_Iris_e200.pt -FD
    """
    main()
