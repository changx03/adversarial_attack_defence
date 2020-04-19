import argparse as ap
import json
import logging
import os
import sys

import numpy as np

from aad.attacks import BIMContainer
from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, get_time_str, master_seed, name_handler
from cmd_utils import get_data_container, parse_model_filename, set_logging
from cross_valid_core import CrossValidation

logger = logging.getLogger('CV')


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='a file which contains a pretrained model. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-p', '--param', type=str, required=True,
        help='a JSON config file which contains the parameters for the cross validation')
    parser.add_argument(
        '-a', '--adv', type=str,
        help='file name of adv. examples for testing. If it\'s none, the program will ignore testing. The name should in "<model>_<dataset>_<attack>_adv.npy" format')
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
        '-i', '--ignore', action='store_true', default=False,
        help='Ignore saving the results. Only returns the results from terminal.')
    parser.add_argument(
        '-w', '--overwrite', action='store_true', default=False,
        help='overwrite the existing file')
    args = parser.parse_args()
    model_file = args.model
    param_file = args.param
    adv_file = args.adv
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog
    does_ignore = args.ignore
    overwrite = args.overwrite

    model_name, data_name = parse_model_filename(model_file)

    # set logging config. Run this before logging anything!
    set_logging('cross_validation', data_name, verbose, save_log)

    # check files
    for file_path in [model_file, param_file]:
        if not os.path.exists(file_path):
            logger.warning('%s does not exist. Exit.', file_path)
            sys.exit(0)
    if adv_file is not None and not os.path.exists(adv_file):
        logger.warning('%s does not exist. Exit.', adv_file)
        sys.exit(0)

    # read parameters
    with open(param_file) as param_json:
        params = json.load(param_json)

    # show parameters
    logger.info('Start at      : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file    :%s', model_file)
    logger.info('adv file      :%s', adv_file)
    logger.info('model         :%s', model_name)
    logger.info('dataset       :%s', data_name)
    logger.info('param file    :%s', param_file)
    logger.info('seed          :%d', seed)
    logger.info('verbose       :%r', verbose)
    logger.info('save_log      :%r', save_log)
    logger.info('Ignore saving :%r', does_ignore)
    logger.info('overwrite     :%r', overwrite)
    logger.info('PARAMETERS FROM JSON FILE:')

    # load parameters
    k_range = params['k_range']
    z_range = params['z_range']
    kappa_range = params['kappa_range']
    gamma_range = params['gamma_range']
    epsilon = params['epsilon']
    num_folds = params['num_folds']
    batch_size = params['batch_size']
    sample_ratio = params['sample_ratio']
    logger.info('k_range       :%s', str(k_range))
    logger.info('z_range       :%s', str(z_range))
    logger.info('kappa_range   :%s', str(kappa_range))
    logger.info('gamma_range   :%s', str(gamma_range))
    logger.info('epsilon       :%.1f', epsilon)
    logger.info('num_folds     :%d', num_folds)
    logger.info('batch_size    :%d', batch_size)
    logger.info('sample_ratio  :%.1f', sample_ratio)

    # reset seed
    master_seed(seed)

    dc = DataContainer(DATASET_LIST[data_name], get_data_path())
    dc(shuffle=True, normalize=True, size_train=0.8)
    logger.info('Sample size: %d', len(dc))

    Model = get_model(model_name)
    # there models require extra keyword arguments
    if data_name in ('BankNote', 'HTRU2', 'Iris', 'WheatSeed'):
        num_classes = dc.num_classes
        num_features = dc.dim_data[0]
        kwargs = {
            'num_features': num_features,
            'hidden_nodes': num_features*4,
            'num_classes': num_classes,
        }
        model = Model(**kwargs)
    else:
        model = Model()
    logger.info('Use %s model', model.__class__.__name__)
    mc = ModelContainerPT(model, dc)
    mc.load(model_file)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    logger.info('Accuracy on test set: %f', accuracy)

    ad = ApplicabilityDomainContainer(
        mc, hidden_model=model.hidden_model, sample_ratio=sample_ratio)
    cross_validation = CrossValidation(
        ad,
        num_folds=num_folds,
        k_range=k_range,
        z_range=z_range,
        kappa_range=kappa_range,
        gamma_range=gamma_range,
        epsilon=epsilon,
    )
    bim_attack = BIMContainer(
        mc,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False,
    )
    cross_validation.fit(bim_attack)

    # test optimal parameters
    if adv_file is not None:
        postfix = ['adv', 'pred', 'x', 'y']
        data_files = [adv_file.replace('_adv', '_' + s) for s in postfix]
        adv = np.load(data_files[0], allow_pickle=False)
        pred_adv = np.load(data_files[1], allow_pickle=False)
        x = np.load(data_files[2], allow_pickle=False)
        pred = np.load(data_files[3], allow_pickle=False)

        # fetch optimal parameters
        k2 = cross_validation.k2
        zeta = cross_validation.zeta
        kappa = cross_validation.kappa
        gamma = cross_validation.gamma
        ad = ApplicabilityDomainContainer(
            mc,
            hidden_model=model.hidden_model,
            k2=k2,
            reliability=zeta,
            sample_ratio=sample_ratio,
            kappa=kappa,
            confidence=gamma
        )
        logger.info('Params: %s', str(ad.params))
        ad.fit()
        blocked_indices = ad.detect(x, pred, return_passed_x=False)
        logger.info('Blocked %d/%d on clean data',
                    len(blocked_indices), len(x))
        blocked_indices = ad.detect(adv, pred_adv, return_passed_x=False)
        logger.info('Blocked %d/%d on adv. examples.',
                    len(blocked_indices), len(adv))

    # save results
    if not does_ignore:
        file_name = name_handler(
            model_name + '_' + data_name, 'csv', overwrite=False)
        cross_validation.save(file_name)


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/cross_valid.py -vl -m ./save/IrisNN_Iris_e200.pt -p ./cmd/CvParams_iris.json -a IrisNN_Iris_Carlini_adv.npy
    $ python ./cmd/cross_valid.py -vl -m ./save/CifarResnet50_CIFAR10_e30.pt -p ./cmd/CvParams_cifar.json
    """
    main()
    print('[attack] Task completed!')
