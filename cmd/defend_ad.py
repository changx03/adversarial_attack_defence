import argparse as ap
import json
import logging
import os

import numpy as np

from aad.basemodels import ModelContainerPT, get_model
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_time_str, master_seed
from cmd_utils import get_data_container, parse_model_filename, set_logging

logger = logging.getLogger('defence')


def detect(detector, set_name, x, y):
    blocked_indices, x_passed = detector.detect(x, y, return_passed_x=True)
    blocked_counts = detector.blocked_by_stages
    logger.info('Blocked %d/%d samples on %s',
                len(blocked_indices), len(x), set_name)
    return x_passed, blocked_indices, blocked_counts.tolist()


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-a', '--adv', type=str, required=True,
        help='file name for adv. examples. The name should in "<model>_<dataset>_<attack>_adv.npy" format')
    parser.add_argument(
        '-p', '--param', type=str, required=True,
        help='a JSON config file which contains the parameters for the attacks')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='a file which contains a pretrained model. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-s', '--seed', type=int, default=4096,
        help='the seed for random number generator')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='set logger level to debug')
    parser.add_argument(
        '-l', '--savelog', action='store_true', default=False,
        help='save logging file')
    args = parser.parse_args()
    adv_file = args.adv
    param_file = args.param
    model_file = args.model
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog
    check_clean = True

    # build filenames from the root file
    postfix = ['adv', 'pred', 'x', 'y']
    data_files = [adv_file.replace('_adv', '_' + s) for s in postfix]
    model_name, dname = parse_model_filename(adv_file)

    # set logging config. Run this before logging anything!
    set_logging('attack', dname, verbose, save_log)

    # check adv. examples and parameter config files
    for f in data_files[:2] + [param_file]:
        if not os.path.exists(f):
            raise FileNotFoundError('{} does not exist!'.format(f))
    # check clean samples
    for f in data_files[-2:]:
        if not os.path.exists(f):
            logger.warning(
                'Cannot load files for clean samples. Skip checking clean set.')
            check_clean = False
    dirname = os.path.dirname(adv_file)

    with open(param_file) as param_json:
        params = json.load(param_json)

    # show parameters
    logger.info('Start at    : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file  :%s', model_file)
    logger.info('adv file    :%s', adv_file)
    logger.info('model       :%s', model_name)
    logger.info('dataset     :%s', dname)
    logger.info('param file  :%s', param_file)
    logger.info('seed        :%d', seed)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('check_clean :%r', check_clean)
    logger.debug('params     : %s', str(params))

    # reset seed
    master_seed(seed)

    # set DataContainer and ModelContainer
    dc = get_data_container(dname)
    Model = get_model(model_name)
    # there models require extra keyword arguments
    if dname in ('BankNote', 'HTRU2', 'Iris', 'WheatSeed'):
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

    # preform defence
    ad = ApplicabilityDomainContainer(mc, hidden_model=model.hidden_model, **params)
    ad.fit()

    result_prefix = [model_file] \
            + [adv_file] \
            + [params['k2']] \
            + [params['reliability']] \
            + [params['sample_ratio']] \
            + [params['confidence']] \
            + [params['kappa']] \
            + [params['disable_s2']]

    # check clean
    if check_clean:
        x = np.load(data_files[2], allow_pickle=False)
        y = np.load(data_files[3], allow_pickle=False)
        x_passed, blk_idx, blocked_counts = detect(ad, 'clean samples', x, y)
        result = result_prefix + ['clean'] + blocked_counts
        result_clean = '[result]' + ','.join([str(r) for r in result])

    # check adversarial examples
    adv = np.load(data_files[0], allow_pickle=False)
    pred = np.load(data_files[1], allow_pickle=False)
    adv_passed, adv_blk_idx, blocked_counts = detect(ad, 'adv. examples', adv, pred)
    result = result_prefix + ['adv'] + blocked_counts
    result = '[result]' + ','.join([str(r) for r in result])
    if check_clean:
        logger.info(result_clean)
    logger.info(result)





if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_FGSM_adv.npy -p ./cmd/AdParams.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt
    $ python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_FGSM_adv.npy -p ./cmd/AdParamsNoS2.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt

    $ python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_DeepFool_adv.npy -p ./cmd/AdParams.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt
    $ python ./cmd/defend_ad.py -vl -a ./save/BCNN_BreastCancerWisconsin_DeepFool_adv.npy -p ./cmd/AdParamsNoS2.json -m ./save/BCNN_BreastCancerWisconsin_e200.pt

    $ python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_Carlini_adv.npy -p ./cmd/AdParamsLarge.json -m ./save/CifarResnet50_SVHN_e30.pt
    $ python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_Carlini_adv.npy -p ./cmd/AdParamsLargeNoS2.json -m ./save/CifarResnet50_SVHN_e30.pt

    $ python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_DeepFool_adv.npy -p ./cmd/AdParamsLarge.json -m ./save/CifarResnet50_SVHN_e30.pt
    $ python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_DeepFool_adv.npy -p ./cmd/AdParamsLargeNoS2.json -m ./save/CifarResnet50_SVHN_e30.pt

    $ python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_FGSM_adv.npy -p ./cmd/AdParamsLarge.json -m ./save/CifarResnet50_SVHN_e30.pt
    $ python ./cmd/defend_ad.py -vl -a ./save/CifarResnet50_SVHN_FGSM_adv.npy -p ./cmd/AdParamsLargeNoS2.json -m ./save/CifarResnet50_SVHN_e30.pt
    """
    main()
    print('[defend_ad] Task completed!')
