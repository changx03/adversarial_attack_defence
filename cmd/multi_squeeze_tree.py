"""
Full pipeline for testing Feature Squeezing (Tree version)

The following steps will repeat 100 times:
1) Build Extra Tree model
2) Generate Adversarial Examples using Decision Tree Attack
3) Train Feature Squeezing
4) Blocking adversarial examples with Feature Squeezing
"""
import argparse as ap
import logging
import os

from art.attacks import DecisionTreeAttack
from art.classifiers import SklearnClassifier
from sklearn.tree import ExtraTreeClassifier

from aad.basemodels import ModelContainerTree
from aad.defences import FeatureSqueezingTree
from aad.utils import get_time_str, name_handler
from cmd_utils import get_data_container, parse_model_filename, set_logging

LOG_NAME = 'MulSqueezeTree'
logger = logging.getLogger(LOG_NAME)

MAX_ITERATIONS = 1
# We don't reset random seed in every run


def experiment(data_name, filter_list, bit_depth, sigma, kernel_size):
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

    # no more than 1000 samples are required
    x = dc.x_test
    y = dc.y_test
    if len(x) > 1000:
        x = x[:1000]
        y = y[:1000]

    accuracy = mc.evaluate(x, y)
    logger.info('Accuracy on clean: %f', accuracy)

    squeezer = FeatureSqueezingTree(
        mc,
        filter_list,
        bit_depth=bit_depth,
        sigma=sigma,
        kernel_size=kernel_size,
        pretrained=True,
    )
    squeezer.fit()
    blocked_indices = squeezer.detect(x)
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
    blocked_indices = squeezer.detect(adv)
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
        '--depth', type=int, default=0,
        help='The image color depth for input images. Apply Binary-Depth filter when receives a parameter')
    parser.add_argument(
        '-s', '--sigma', type=float, default=0,
        help='The Standard Deviation of Normal distribution. Apply Gaussian Noise filter when receives a parameter')
    parser.add_argument(
        '-k', '--kernelsize', type=int, default=0,
        help='The kernel size for Median filter. Apply median filter when receives a parameter')
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
    args = parser.parse_args()
    data_name = args.dataset
    max_iterations = args.iteration
    bit_depth = args.depth
    sigma = args.sigma
    kernel_size = args.kernelsize
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # set logging config. Run this before logging anything!
    set_logging(LOG_NAME, data_name, verbose, save_log)

    # Which filter should apply?
    filter_list = []
    if bit_depth > 0:
        filter_list.append('binary')
    if sigma > 0:
        filter_list.append('normal')
    if kernel_size > 0:
        filter_list.append('median')

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
    logger.info('bit_depth   :%d', bit_depth)
    logger.info('sigma       :%f', sigma)
    logger.info('kernel_size :%d', kernel_size)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('overwrite   :%r', overwrite)
    logger.info('filename    :%s', result_filename)

    # NOTE: Why does train all adversarial examples not work?
    # The classification models are depended on the training set. They are not
    # identical, thus adversarial examples are also not the same.

    with open(result_filename, 'w') as file:
        file.write(','.join(['Index', 'Clean', 'DecisionTreeAttack']) + '\n')
        for i in range(max_iterations):
            num_blk_clean, num_blk_adv = experiment(
                data_name, filter_list, bit_depth, sigma, kernel_size)
            file.write(f'{i},{num_blk_clean},{num_blk_adv}\n')
        file.close()


# Examples:
# python ./cmd/multi_squeeze_tree.py -vl -i 1 -d Iris --depth 8 -s 0.2
# python ./cmd/multi_squeeze_tree.py -vl -i 1 -d BankNote --depth 8 -s 0.2
# python ./cmd/multi_squeeze_tree.py -vl -i 1 -d WheatSeed --depth 8 -s 0.2
# python ./cmd/multi_squeeze_tree.py -vl -i 1 -d HTRU2 --depth 8 -s 0.2
# python ./cmd/multi_squeeze_tree.py -vl -i 1 -d BreastCancerWisconsin --depth 8 -s 0.2
if __name__ == '__main__':
    main()
    print(f'[{LOG_NAME}] Task completed!')
