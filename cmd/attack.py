import argparse as ap
import json
import logging
import os
import sys

import numpy as np

from aad.attacks import get_attack
from aad.basemodels import ModelContainerPT, get_model
from aad.utils import get_time_str, master_seed
from cmd_utils import get_data_container, parse_model_filename, set_logging

logger = logging.getLogger('attack')


def run_attacks(model_container,
                selected_attacks,
                params,
                count,
                filename,
                overwrite):
    """Run selected adversarial attacks"""
    for att_name in selected_attacks:
        adv_filename = filename + '_' + att_name
        Attack = get_attack(att_name)
        kwargs = params[att_name]
        logger.debug('%s params: %s', att_name, str(kwargs))
        attack = Attack(model_container, **kwargs)
        adv, y_adv, x_clean, y_clean = attack.generate(count=count)
        logger.info('# of adv created from %s: %d', filename, len(adv))
        not_match = y_adv != y_clean
        success_rate = len(not_match[not_match == True]) / len(adv)
        accuracy = model_container.evaluate(adv, y_clean)
        logger.info('Success rate of %s: %f', att_name, success_rate)
        logger.info('Accuracy on %s: %f', att_name, accuracy)
        logger.debug('Save adv. attack results into: %s', adv_filename)
        attack.save_attack(adv_filename, adv, y_adv,
                           x_clean, y_clean, overwrite)


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='a file which contains a pretrained model. The filename should in "<model>_<dataset>_e<max epochs>[_<date>].pt" format')
    parser.add_argument(
        '-p', '--param', type=str, required=True,
        help='a JSON config file which contains the parameters for the attacks')
    parser.add_argument(
        '-n', '--number', type=int, default=100,
        help='the number of adv. examples want to generate. (if more than test set, it uses all test examples.)')
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
        '-w', '--overwrite', action='store_true', default=False,
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
    model_file = args.model
    attack_param_file = args.param
    num_adv = args.number
    seed = args.seed
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # Which attack should apply?
    # use binary encoding for attacks
    my_attacks = np.zeros(5, dtype=np.int8)
    attack_list = np.array(
        ['FGSM', 'BIM', 'Carlini', 'DeepFool', 'Saliency'])
    my_attacks[0] = 1 if args.fgsm else 0
    my_attacks[1] = 1 if args.bim else 0
    my_attacks[2] = 1 if args.carlini else 0
    my_attacks[3] = 1 if args.deepfool else 0
    my_attacks[4] = 1 if args.saliency else 0
    selected_attacks = attack_list[np.where(my_attacks == 1)[0]]

    # check file
    for f in [model_file, attack_param_file]:
        if not os.path.exists(f):
            raise FileNotFoundError('{} does not exist!'.format(f))
    dirname = os.path.dirname(model_file)
    model_name, dname = parse_model_filename(model_file)

    with open(attack_param_file) as param_json:
        att_params = json.load(param_json)

    # set logging config. Run this before logging anything!
    set_logging('attack', dname, verbose, save_log)

    # show parameters
    logger.info('Start at   : %s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('model file :%s', model_file)
    logger.info('model      :%s', model_name)
    logger.info('dataset    :%s', dname)
    logger.info('params     :%s', attack_param_file)
    logger.info('num_adv    :%r', num_adv)
    logger.info('seed       :%d', seed)
    logger.info('verbose    :%r', verbose)
    logger.info('save_log   :%r', save_log)
    logger.info('overwrite  :%r', overwrite)
    logger.info('dirname    :%r', dirname)
    logger.info('attacks    :%s', ', '.join(selected_attacks))

    if len(selected_attacks) == 0:
        logger.warning('No attack is selected. Exit.')
        sys.exit(0)

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

    run_attacks(mc,
                selected_attacks,
                att_params,
                num_adv,
                model_name + '_' + dname,
                overwrite)


if __name__ == '__main__':
    """
    Examples:
    $ python ./cmd/attack.py -m ./save/BCNN_BreastCancerWisconsin_e200.pt -p ./cmd/AttackParams.json -lvw -FD
    $ python ./cmd/attack.py -m ./save/CifarCnn_CIFAR10_e50.pt -p ./cmd/AttackParams.json -vw -FD
    $ python ./cmd/attack.py -m ./save/IrisNN_BankNote_e200.pt -p ./cmd/AttackParams.json -vw -FD
    $ python ./cmd/attack.py -m ./save/IrisNN_HTRU2_e200.pt -p ./cmd/AttackParams.json -vw -FD
    $ python ./cmd/attack.py -m ./save/IrisNN_Iris_e200.pt -p ./cmd/AttackParams.json -vw -FD
    $ python ./cmd/attack.py -m ./save/IrisNN_WheatSeed_e300.pt -p ./cmd/AttackParams.json -vw -FD
    $ python ./cmd/attack.py -m ./save/MnistCnnV2_MNIST_e50.pt -p ./cmd/AttackParams.json -vw -FD
    $ python ./cmd/attack.py -m ./save/CifarResnet50_SVHN_e30.pt -p ./cmd/AttackParams.json -vw -F
    $ python ./cmd/attack.py -m ./save/CifarResnet50_SVHN_e30.pt -p ./cmd/AttackParams.json -vw -D
    $ python ./cmd/attack.py -m ./save/CifarResnet50_SVHN_e30.pt -p ./cmd/AttackParams.json -w -C
    """
    main()
    print('[attack] Task completed!')
