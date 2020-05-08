"""
Full pipeline for testing adversarial defences

The following steps will repeat 100 times:
1) Build and train the model
2) Generate adversarial examples
3) Train the defensive model
4) Blocking adversarial examples with the defensive model
"""
import argparse as ap
import json
import logging
import os
import sys

import numpy as np

from aad.attacks import get_attack
from aad.basemodels import ModelContainerPT, get_model
from aad.utils import get_time_str, master_seed, name_handler
from cmd_utils import get_data_container, parse_model_filename, set_logging

LOG_NAME = 'Mul'
logger = logging.getLogger(LOG_NAME)

# We don't reset random seed in every run
MAX_ITERATIONS = 100
TITLE = [
    'Clean:AdvTraining',
    'Clean:Destillation',
    'Clean:Squeezing',
    'Clean:AD',
    'FGSM:AdvTraining',
    'FGSM:Destillation',
    'FGSM:Squeezing',
    'FGSM:AD',
    'BIM:AdvTraining',
    'BIM:Destillation',
    'BIM:Squeezing',
    'BIM:AD',
    'DeepFool:AdvTraining',
    'DeepFool:Destillation',
    'DeepFool:Squeezing',
    'DeepFool:AD',
    'C&W:AdvTraining',
    'C&W:Destillation',
    'C&W:Squeezing',
    'C&W:AD',
]
ATTACK_LIST = ['FGSM', 'BIM', 'DeepFool', 'Carlini']


def experiment(index, dname, file):
    pass


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help='Name of the dataset')
    parser.add_argument(
        '-i', '--iteration', type=int, default=MAX_ITERATIONS,
        help='the number of iterations that the experiment will repeat')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='set logger level to debug')
    parser.add_argument(
        '-l', '--savelog', action='store_true', default=False,
        help='save logging file')
    parser.add_argument(
        '-w', '--overwrite', action='store_true', default=False,
        help='overwrite the existing file')

    # NOTE: the JSON file for parameter are hard coded.
    # We expect to run multiple attacks and defences in one iteration.
    args = parser.parse_args()
    dname = args.dataset
    max_iterations = args.iteration
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # set logging config. Run this before logging anything!
    set_logging(LOG_NAME, dname, verbose, save_log)

    print('[{}] Start experiment on {}...'.format(LOG_NAME, dname))
    logger.info('Start at      : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('dataset    :%s', dname)
    logger.info('iterations  :%d', max_iterations)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('overwrite   :%r', overwrite)

    result_file = name_handler(
        os.path.join('save', f'{LOG_NAME}_{dname}_i{max_iterations}'),
        'csv',
        overwrite=overwrite
    )

    with open(result_file, 'w') as file:
        file.write(','.join(TITLE) + '\n')
        for i in range(max_iterations):
            experiment(i, dname, file)
        file.close()


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/multi_numeric.py -l -d Iris
    """
    main()
    print(f'[{LOG_NAME}] Task completed!')
