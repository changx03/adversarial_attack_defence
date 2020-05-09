import argparse as ap
import logging
import os
import sys

import numpy as np

from aad.basemodels import IrisNN, ModelContainerPT, get_model
from aad.defences import FeatureSqueezing
from aad.utils import get_time_str, master_seed
from cmd_utils import get_data_container, parse_model_filename, set_logging

LOG_NAME = 'DefSqueeze'
logger = logging.getLogger(LOG_NAME)


def build_squeezer_filename(model_name, data_name, max_epochs, filter_name):
    """
    Pre-train file example: MnistCnnV2_MNIST_e50_binary.pt
    """
    return f'{model_name}_{data_name}_e{max_epochs}_{filter_name}.pt'


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='a file which contains a pretrained model. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-e', '--epoch', type=int, required=True,
        help='the number of max epochs for training')
    parser.add_argument(
        '-d', '--depth', type=int, default=0,
        help='The image color depth for input images. Apply Binary-Depth filter when receives a parameter')
    parser.add_argument(
        '--sigma', type=float, default=0,
        help='The Standard Deviation of Normal distribution. Apply Gaussian Noise filter when receives a parameter')
    parser.add_argument(
        '-k', '--kernelsize', type=int, default=0,
        help='The kernel size for Median filter. Apply median filter when receives a parameter')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=128, help='batch size')
    parser.add_argument(
        '-T', '--train', action='store_true', default=False,
        help='Force the model to retrain without searching existing pretrained file')
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
    bit_depth = args.depth
    sigma = args.sigma
    kernel_size = args.kernelsize
    batch_size = args.batchsize
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog
    need_train = args.train

    model_name, data_name = parse_model_filename(model_file)

    # Which filter should apply?
    filter_list = []
    if bit_depth > 0:
        filter_list.append('binary')
    if sigma > 0:
        filter_list.append('normal')
    if kernel_size > 0:
        filter_list.append('median')

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
    if args.saliency:
        attack_list.append('Saliency')

    # Quit, if there is nothing to do.
    if len(filter_list) == 0 or len(attack_list) == 0:
        logger.warning('Neither received any filter nor any attack. Exit')
        sys.exit(0)

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

    # Do I need train the distillation network?
    pretrain_files = []
    for fname in filter_list:
        pretrain_file = build_squeezer_filename(
            model_name, data_name, max_epochs, fname
        )
        pretrain_files.append(pretrain_file)
        if not os.path.exists(os.path.join('save', pretrain_file)):
            need_train = True

    # set logging config. Run this before logging anything!
    set_logging(LOG_NAME, data_name, verbose, save_log)

    # show parameters
    print(f'[{LOG_NAME}] Running feature squeezing on {model_name}...')
    logger.info('Start at    : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file  :%s', model_file)
    logger.info('model       :%s', model_name)
    logger.info('dataset     :%s', data_name)
    logger.info('max_epochs  :%d', max_epochs)
    logger.info('bit_depth   :%d', bit_depth)
    logger.info('sigma       :%f', sigma)
    logger.info('kernel_size :%d', kernel_size)
    logger.info('batch_size  :%d', batch_size)
    logger.info('seed        :%d', seed)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('need train  :%r', need_train)
    logger.info('filters     :%s', ', '.join(filter_list))
    logger.info('attacks     :%s', ', '.join(attack_list))
    logger.info('pretrained  :%s', ', '.join(pretrain_files))

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

    # initialize Squeezer
    squeezer = FeatureSqueezing(
        classifier_mc,
        filter_list,
        bit_depth=bit_depth,
        sigma=sigma,
        kernel_size=kernel_size,
        pretrained=True,
    )

    # train or load parameters for Squeezer
    if need_train:
        squeezer.fit(max_epochs=max_epochs, batch_size=batch_size)
        squeezer.save(model_file, True)
    else:
        squeezer.load(model_file)

    # traverse all attacks
    y = np.load(y_file, allow_pickle=False)
    for i in range(len(attack_list)):
        adv_file = attack_files[i]
        adv_name = attack_list[i]
        logger.debug('Load %s...', adv_file)
        adv = np.load(adv_file, allow_pickle=False)
        acc_og = classifier_mc.evaluate(adv, y)
        acc_squeezer = squeezer.evaluate(adv, y)
        logger.info('Accuracy on %s set - OG: %f, Squeezer: %f',
                    adv_name, acc_og, acc_squeezer)
        blocked_indices = squeezer.detect(adv, return_passed_x=False)
        logger.info('Blocked %d/%d samples on %s',
                    len(blocked_indices), len(adv), adv_name)


# Examples:
# python ./cmd/defend_squeeze.py -v -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/MnistCnnV2_MNIST_e50.pt -FBDCS
# python ./cmd/defend_squeeze.py -v -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/CifarCnn_CIFAR10_e50.pt -FBDCS
# python ./cmd/defend_squeeze.py -v -e 50 -d 8 --sigma 0.2 -k 3 -m ./save/CifarCnn_SVHN_e50.pt -FBDCS
if __name__ == '__main__':
    main()
    print(f'[{LOG_NAME}] Task completed!')
