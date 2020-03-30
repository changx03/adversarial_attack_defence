import argparse as ap
import json
import logging
import os

from aad.basemodels import ModelContainerPT, get_model
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_time_str, master_seed
from cmd_utils import get_data_container, parse_model_filename, set_logging

logger = logging.getLogger('defence')


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='a file which contains a pretrained model. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-a', '--adv', type=str, required=True,
        help='file name for adversarial examples. The name should in "<model>_<dataset>_<attack>_adv.npy" format')
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
    args = parser.parse_args()
    model_file = args.model
    filename = args.adv
    param_file = args.param
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog

    # build filenames from the root file
    postfix = ['adv', 'pred', 'x', 'y']
    data_files = [filename.replace('_adv', '_' + s) for s in postfix]
    print(data_files)

    # check file
    for f in data_files + [param_file]:
        if not os.path.exists(f):
            raise FileNotFoundError('{} does not exist!'.format(f))
    dirname = os.path.dirname(filename)
    model_name, dname = parse_model_filename(filename)

    with open(param_file) as param_json:
        params = json.load(param_json)

    # set logging config. Run this before logging anything!
    set_logging('attack', dname, verbose, save_log)

    # show parameters
    logger.info('Start at   : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file :%s', model_file)
    logger.info('adv file   :%s', filename)
    logger.info('model      :%s', model_name)
    logger.info('dataset    :%s', dname)
    logger.info('param file :%s', param_file)
    logger.info('seed       :%d', seed)
    logger.info('verbose    :%r', verbose)
    logger.info('save_log   :%r', save_log)
    logger.debug('params    : %s', str(params))

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
    accuracy = mc.evaluate(dc.data_test_np, dc.label_test_np)
    logger.info('Accuracy on test set: %f', accuracy)

    # preform defence
    ad = ApplicabilityDomainContainer(mc, **params)
    ad.fit()

    # TODO: test on clean samples, test on adv samples


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/defend_ad.py \
        -m ./save/BCNN_BreastCancerWisconsin_e200.pt \
        -a ./save/BCNN_BreastCancerWisconsin_FGSM_adv.npy \
        -p ./cmd/ad_params.json -v
    """
    main()
