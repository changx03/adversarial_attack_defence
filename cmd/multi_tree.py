"""
Full pipeline for testing Applicability Domain

The following steps will repeat 100 times:
1) Build Extra Tree model
2) Generate Adversarial Examples using Decision Tree Attack
3) Train Applicability Domain
4) Blocking adversarial examples with Applicability Domain
"""
import argparse as ap
import json
import logging
import os

from art.attacks import DecisionTreeAttack
from art.classifiers import SklearnClassifier
from sklearn.tree import ExtraTreeClassifier

from aad.basemodels import ModelContainerTree
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_time_str, name_handler
from cmd_utils import get_data_container, set_logging

LOG_NAME = 'MulTree'
logger = logging.getLogger(LOG_NAME)

MAX_ITERATIONS = 100
# We don't reset random seed in every run


def experiment(data_name, params):
    # select data
    dc = get_data_container(
        data_name,
        use_shuffle=True,
        use_normalize=True,
    )

    # train the model
    # train the model
    classifier = ExtraTreeClassifier(
        criterion='gini',
        splitter='random',
    )
    mc = ModelContainerTree(classifier, dc)
    mc.fit()

    # train Applicability Domain
    ad = ApplicabilityDomainContainer(
        mc, mc.hidden_model, **params)
    ad.fit()

    # no more than 1000 samples are required
    x = dc.x_test
    y = dc.y_test
    if len(x) > 1000:
        x = x[:1000]
        y = y[:1000]

    accuracy = mc.evaluate(x, y)
    logger.info('Accuracy on clean: %f', accuracy)
    blocked_indices = ad.detect(x)
    logger.info('Blocked %d/%d samples on clean',
                len(blocked_indices), len(y))
    num_blk_clean = len(blocked_indices)

    # generate adversarial examples
    art_classifier = SklearnClassifier(classifier)
    try:
        attack = DecisionTreeAttack(art_classifier)
        adv = attack.generate(x)
    except IndexError as error:
        # Output expected IndexErrors.
        logger.error(error)
        return num_blk_clean, -1

    accuracy = mc.evaluate(adv, y)
    logger.info('Accuracy on DecisionTreeAttack: %f', accuracy)
    blocked_indices = ad.detect(adv)
    logger.info('Blocked %d/%d samples on DecisionTreeAttack',
                len(blocked_indices), len(adv))
    num_blk_adv = len(blocked_indices)
    return num_blk_clean, num_blk_adv


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help='Name of the dataset')
    parser.add_argument(
        '-i', '--iteration', type=int, default=MAX_ITERATIONS,
        help='the number of iterations that the experiment will repeat')
    parser.add_argument(
        '-p', '--param', type=str, required=True,
        help='a JSON config file which contains the parameters for the applicability domain')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='set logger level to debug')
    parser.add_argument(
        '-l', '--savelog', action='store_true', default=False,
        help='save logging file')
    parser.add_argument(
        '-w', '--overwrite', action='store_true', default=False,
        help='overwrite the existing file')
    args = parser.parse_args()
    data_name = args.dataset
    max_iterations = args.iteration
    param_file = args.param
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # set logging config. Run this before logging anything!
    set_logging(LOG_NAME, data_name, verbose, save_log)

    # load parameters for Applicability Domain
    with open(param_file) as param_json:
        params = json.load(param_json)

    result_filename = name_handler(
        os.path.join('save', LOG_NAME + '_' + data_name + '_' + 'tree'),
        'csv',
        overwrite=overwrite)

    # show parameters
    print(f'[{LOG_NAME}] Running tree model...')
    logger.info('Start at    :%s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('dataset     :%s', data_name)
    logger.info('iterations  :%d', max_iterations)
    logger.info('param file  :%s', param_file)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('overwrite   :%r', overwrite)
    logger.info('filename    :%s', result_filename)
    logger.debug('params     :%s', str(params))

    # NOTE: Why does train all adversarial examples not work?
    # The classification models are depended on the training set. They are not
    # identical, thus adversarial examples are also not the same.

    with open(result_filename, 'w') as file:
        file.write(','.join(['index', 'clean', 'DecisionTreeAttack']) + '\n')
        for i in range(max_iterations):
            num_blk_clean, num_blk_adv = experiment(data_name, params)
            file.write(f'{i},{num_blk_clean},{num_blk_adv}\n')
        file.close()


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d Iris
    $ python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d BankNote
    $ python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d WheatSeed
    $ python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d HTRU2
    $ python ./cmd/multi_tree.py -vl -p ./cmd/AdParamsNumeral.json -d BreastCancerWisconsin
    """
    main()
    print(f'[{LOG_NAME}] Task completed!')
