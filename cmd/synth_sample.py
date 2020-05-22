"""
This module is the full pipeline for testing synthetic data with controlled
parameters.

The following steps will repeat 30 times:
1) Build and train the model
2) Generate adversarial examples
3) Train the defensive model
4) Blocking adversarial examples with the defensive model

Experiment 1: Binary classification, control # of features
    # of classes = 2 (Binary classification)
    # of features = 30 (Same as Breast Cancer Wisconsin)

    Variation:
        Sample size = 250, 500, 1000, 5000, 10000, 50000

Experiment 2: Binary classification, control sample size
    # of classes = 2
    # Sample size = 5000

    Variation:
        # of features = 4, 8, 16, 32, 64, 128, 256, 512 (double every time)

Experiment 3: Binary classification, control sample_size / num_features ratio
    # of classes = 2

    Variation:
        Ratio | Sample Size | # of features
        20    |         500 |            25
        20    |        1000 |            50
        20    |       10000 |           500
        40    |         500 |            12
        40    |        1000 |            25
        40    |       10000 |           250
        80    |         500 |             6
        80    |        1000 |            12
        80    |       10000 |           120
"""
import argparse as ap
import json
import logging
import os
import time

import numpy as np
from sklearn.datasets import make_classification

from aad.attacks import BIMContainer, get_attack
from aad.basemodels import IrisNN, ModelContainerPT
from aad.datasets import DataContainer, get_synthetic_dataset_dict
from aad.defences import (AdversarialTraining, ApplicabilityDomainContainer,
                          DistillationContainer, FeatureSqueezing)
from aad.utils import (get_data_path, get_time_str, name_handler,
                       scale_normalize)
from cmd_utils import get_data_container, set_logging

LOG_NAME = 'SynthSample'
logger = logging.getLogger(LOG_NAME)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# We don't reset random seed in every run
# We run 1 iteration at a time, so multiple job can work parallel.
MAX_ITERATIONS = 1
BATCH_SIZE = 128  # 250*0.8
TITLE_RESULTS = [
    'Index',
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
ATTACK_LIST = ['Clean', 'FGSM', 'BIM', 'DeepFool', 'Carlini']
TITLE_ADV = ['Test'] + ATTACK_LIST
DEFENCE_LIST = ['AdvTraining', 'Destillation', 'Squeezing', 'AD']
ADV_TRAIN_RATIO = 0.25
SQUEEZER_FILTER_LIST = ['binary', 'normal']
SQUEEZER_DEPTH = 8
SQUEEZER_SIGMA = 0.2
AD_PARAM_FILE = os.path.join(DIR_PATH, 'AdParamsNumeral.json')


def block_attack(offset, advs, defence, def_name, blocked_res):
    for j in range(len(ATTACK_LIST)):
        adv = advs[j]
        blocked_indices = defence.detect(adv, return_passed_x=False)
        blocked_res[j*len(DEFENCE_LIST) + offset + 1] = len(blocked_indices)
        logger.info('%s blocks %d/%d samples on %s',
                    def_name, len(blocked_indices), len(adv), ATTACK_LIST[j])


def experiment(index, data_container, max_epochs, adv_file, res_file):
    # STEP 1: select data and model
    dname = data_container.name
    num_classes = data_container.num_classes
    num_features = data_container.dim_data[0]
    model = IrisNN(
        num_features=num_features,
        hidden_nodes=num_features*4,
        num_classes=num_classes
    )
    distill_model = IrisNN(
        num_features=num_features,
        hidden_nodes=num_features*4,
        num_classes=num_classes
    )

    # STEP 2: train models
    mc = ModelContainerPT(model, data_container)
    mc.fit(max_epochs=max_epochs, batch_size=BATCH_SIZE)
    accuracy = mc.evaluate(data_container.x_test, data_container.y_test)
    logger.info('Accuracy on test set: %f', accuracy)
    adv_res = [accuracy]

    # STEP 3: generate adversarial examples
    # no more than 1000 samples are required
    x = data_container.x_test
    y = data_container.y_test
    # The test set has fixed size, 1000.
    if len(x) > 1000:
        x = x[:1000]
        y = y[:1000]
    accuracy = mc.evaluate(x, y)
    adv_res.append(accuracy)

    advs = np.zeros((len(ATTACK_LIST), x.shape[0], x.shape[1]),
                    dtype=np.float32)
    pred_advs = -np.ones((len(ATTACK_LIST), len(y)),
                         dtype=np.int32)  # assign -1 as initial value
    pred_clean = mc.predict(x)

    advs[0] = x
    pred_advs[0] = pred_clean

    att_param_json = open(os.path.join(DIR_PATH, 'AttackParams.json'))
    att_params = json.load(att_param_json)

    for i, att_name in enumerate(ATTACK_LIST):
        # Clean set is only used in evaluation phase.
        if att_name == 'Clean':
            continue

        logger.debug('[%d]Running %s attack...', i, att_name)
        kwargs = att_params[att_name]
        logger.debug('%s params: %s', att_name, str(kwargs))
        Attack = get_attack(att_name)
        attack = Attack(mc, **kwargs)
        adv, pred_adv, x_clean, pred_clean_ = attack.generate(
            use_testset=False,
            x=x)
        assert np.all(pred_clean == pred_clean_)
        assert np.all(x == x_clean)
        logger.info('created %d adv examples using %s from %s',
                    len(advs[i]),
                    att_name,
                    dname)
        not_match = pred_adv != pred_clean
        success_rate = len(not_match[not_match == True]) / len(pred_clean)
        accuracy = mc.evaluate(adv, y)
        advs[i] = adv
        pred_advs[i] = pred_adv
        logger.info('Success rate of %s: %f', att_name, success_rate)
        logger.info('Accuracy on %s: %f', att_name, accuracy)
        adv_res.append(accuracy)
    adv_file.write(','.join([str(r) for r in adv_res]) + '\n')

    # STEP 4: train defences
    blocked_res = np.zeros(len(TITLE_RESULTS), dtype=np.int32)
    blocked_res[0] = index
    for def_name in DEFENCE_LIST:
        logger.debug('Running %s...', def_name)
        if def_name == 'AdvTraining':
            attack = BIMContainer(
                mc,
                eps=0.3,
                eps_step=0.1,
                max_iter=100,
                targeted=False)
            defence = AdversarialTraining(mc, [attack])
            defence.fit(max_epochs=max_epochs,
                        batch_size=BATCH_SIZE,
                        ratio=ADV_TRAIN_RATIO)
            block_attack(0, advs, defence, def_name, blocked_res)
        elif def_name == 'Destillation':
            # A single temperature is used for all sets
            temp = 20
            defence = DistillationContainer(
                mc, distill_model, temperature=temp, pretrained=False)
            defence.fit(max_epochs=max_epochs, batch_size=BATCH_SIZE)
            block_attack(1, advs, defence, def_name, blocked_res)
        elif def_name == 'Squeezing':
            defence = FeatureSqueezing(
                mc,
                SQUEEZER_FILTER_LIST,
                bit_depth=SQUEEZER_DEPTH,
                sigma=SQUEEZER_SIGMA,
                pretrained=True,
            )
            defence.fit(max_epochs=max_epochs, batch_size=BATCH_SIZE)
            block_attack(2, advs, defence, def_name, blocked_res)
        elif def_name == 'AD':
            ad_param_file = open(AD_PARAM_FILE)
            # BreastCancer uses a different set of parameters
            if dname == 'BreastCancerWisconsin':
                param_file = os.path.join(DIR_PATH, 'AdParamsBC.json')
                ad_param_file = open(param_file)
            ad_params = json.load(ad_param_file)
            logger.debug('AD params: %s', str(ad_params))
            defence = ApplicabilityDomainContainer(
                mc,
                hidden_model=model.hidden_model,
                **ad_params)
            defence.fit()
            block_attack(3, advs, defence, def_name, blocked_res)

    res_file.write(','.join([str(r) for r in blocked_res]) + '\n')


def main():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-s', '--size', type=int, required=True,
        help='the number of sample size')
    parser.add_argument(
        '-f', '--features', type=int, required=True,
        help='the number of features')
    parser.add_argument(
        '-c', '--classes', type=int, default=2,
        help='the number of classes')
    parser.add_argument(
        '-i', '--iteration', type=int, default=MAX_ITERATIONS,
        help='the number of iterations that the experiment will repeat')
    parser.add_argument(
        '-e', '--epoch', type=int, required=True,
        help='the number of max epochs for training')
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
    sample_size = args.size
    num_features = args.features
    num_classes = args.classes
    max_iterations = args.iteration
    max_epochs = args.epoch
    verbose = args.verbose
    save_log = args.savelog
    overwrite = args.overwrite

    # set logging config. Run this before logging anything!
    dname = f'SyntheticS{sample_size}F{num_features}C{num_classes}'
    set_logging(LOG_NAME, dname, verbose, save_log)

    print('[{}] Start experiment on {}...'.format(LOG_NAME, dname))
    logger.info('Start at    :%s', get_time_str())
    logger.info('RECEIVED PARAMETERS:')
    logger.info('dataset     :%s', dname)
    logger.info('train size  :%d', sample_size)
    logger.info('num features:%d', num_features)
    logger.info('num classes :%d', num_classes)
    logger.info('iterations  :%d', max_iterations)
    logger.info('max_epochs  :%d', max_epochs)
    logger.info('verbose     :%r', verbose)
    logger.info('save_log    :%r', save_log)
    logger.info('overwrite   :%r', overwrite)

    result_file = name_handler(
        os.path.join('save', f'{LOG_NAME}_{dname}_i{max_iterations}'),
        'csv',
        overwrite=overwrite
    )

    adv_file = name_handler(
        os.path.join('save', f'{LOG_NAME}_{dname}_AdvExamples'),
        'csv',
        overwrite=overwrite
    )

    adv_file = open(adv_file, 'w')
    adv_file.write(','.join(TITLE_ADV) + '\n')
    res_file = open(result_file, 'w')
    res_file.write(','.join(TITLE_RESULTS) + '\n')
    for i in range(max_iterations):
        since = time.time()
        # generate synthetic data
        x, y = make_classification(
            n_samples=sample_size+1000,
            n_features=num_features,
            n_informative=num_classes,
            n_redundant=0,
            n_classes=num_classes,
            n_clusters_per_class=1,
        )

        # normalize data
        x_max = np.max(x, axis=0)
        x_min = np.min(x, axis=0)
        # NOTE: Carlini attack expects the data in range [0, 1]
        # x_mean = np.mean(x, axis=0)
        # x = scale_normalize(x, x_min, x_max, x_mean)
        x = scale_normalize(x, x_min, x_max)

        # training/test split
        # NOTE: test set has fixed size
        x_train = np.array(x[:-1000], dtype=np.float32)
        y_train = np.array(y[:-1000], dtype=np.long)
        x_test = np.array(x[-1000:], dtype=np.float32)
        y_test = np.array(y[-1000:], dtype=np.long)

        # create data container
        data_dict = get_synthetic_dataset_dict(
            sample_size+1000, num_classes, num_features)
        dc = DataContainer(data_dict, get_data_path())

        # assign data manually
        dc.x_train = x_train
        dc.y_train = y_train
        dc.x_test = x_test
        dc.y_test = y_test

        experiment(i, dc, max_epochs, adv_file, res_file)
        time_elapsed = time.time() - since
        print('Completed {} [{}/{}]: {:d}m {:2.1f}s'.format(
            dname,
            i+1,
            max_iterations,
            int(time_elapsed // 60),
            time_elapsed % 60))

    adv_file.close()
    res_file.close()

# Examples:
# Experiment 1:
# python ./cmd/synth_sample.py -vl -s 250 -f 30 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 500 -f 30 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 1000 -f 30 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 30 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 10000 -f 30 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 50000 -f 30 -c 2 -i 1 -e 200

# Experiment 2:
# python ./cmd/synth_sample.py -vl -s 5000 -f 4 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 8 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 16 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 32 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 64 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 128 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 256 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 5000 -f 512 -c 2 -i 1 -e 200

# Experiment 3:
# python ./cmd/synth_sample.py -vl -s 500 -f 25 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 1000 -f 50 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 10000 -f 500 -c 2 -i 1 -e 200

# python ./cmd/synth_sample.py -vl -s 500 -f 12 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 1000 -f 25 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 10000 -f 250 -c 2 -i 1 -e 200


# python ./cmd/synth_sample.py -vl -s 500 -f 6 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 1000 -f 12 -c 2 -i 1 -e 200
# python ./cmd/synth_sample.py -vl -s 10000 -f 120 -c 2 -i 1 -e 200
if __name__ == '__main__':
    main()
    print(f'[{LOG_NAME}] Task completed!')
