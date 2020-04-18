"""
This module implements linear search for best parameters for Applicability Domain.
"""
import logging
import os
import time

import numpy as np
import torch

import aad.attacks as attacks
from aad.basemodels import MnistCnnV2, ModelContainerPT
from aad.datasets import DATASET_LIST, CustomDataContainer, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import cross_validation_split, get_data_path, master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEED = 4096
BATCH_SIZE = 64
NAME = 'MNIST'
SAMPLE_RATIO = 1.0 / 6.0
MAX_EPOCHS = 50
MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')
CARLINI_FILE = os.path.join('save', 'MnistCnnV2_MNIST_Carlini_adv.npy')

# initial parameters
K2 = 2  # best k2 = 2, found by linear search
# parameter zeta. The idea comes from confidence interval
RELIABILITY = 1.8  # best zeta = 1.8, found by linear search
# No huge improvement on the searching space
KAPPA = 10
CONFIDENCE = 1.0

K_RANGE = (1, 32)
Z_RANGE = (1.0, 3.0)
KAPPA_RANGE = (5, 24)
GAMMA_RANGE = (0.5, 1.0)
# weight for penalting blocking clean data
# score = -blocked_clean + (epsilon * blocked_adv.)
# If epsilon > 1, blocking adversarial examples is more important than not blockling clean samples.
# If epsilon < 1, emphasising preserve the clean set rather than blocking adversarial examples.
# If epsilon = 1, both are weight the same.
EPSILON = 1.0
# Only consider the parameter when 90% of the clean samples can pass the test.
PASS_THRESHOLD_S2 = 0.9

NUM_FOLDS = 5


def update_k2(ad, update, const):
    logger.debug('Received k2: %f, zeta: %f', update, const)
    ad.fit_stage2_(update, const)


def update_zeta(ad, update, const):
    logger.debug('Received k2: %f, zeta: %f', const, update)
    ad.fit_stage2_(const, update)


def def_stage2(ad, encoded_data, labels, passed):
    return ad.def_stage2_(encoded_data, labels, passed)


def update_kappa(ad, update, const):
    logger.debug('Received kappa: %f, gamma: %f', update, const)
    ad.fit_stage3_(update, const)


def update_gamma(ad, update, const):
    logger.debug('Received kappa: %f, gamma: %f', const, update)
    ad.fit_stage3_(const, update)


def def_stage3(ad, encoded_data, labels, passed):
    return ad.def_stage3_(encoded_data, labels, passed)


def search_param(ad, update_fn, def_stage, drange, increment, const_param,
                 adv, pred_adv, x_clean, y_true):
    i = drange[0]
    max_score = 0
    best = i
    scores_clean = []
    scores_adv = []
    params = []
    n = len(adv)
    while i <= drange[1]:
        update_fn(ad, i, const_param)
        passed = np.ones(n, dtype=np.int8)
        encoded_clean = ad.preprocessing_(x_clean)
        passed_clean = def_stage(ad, encoded_clean, y_true, passed)
        score_clean = len(passed_clean[passed_clean == 0])
        encoded_adv = ad.preprocessing_(adv)
        passed_adv = def_stage(ad, encoded_adv, pred_adv, passed)
        score_adv = len(passed_clean[passed_adv == 0])
        score = -score_clean + EPSILON * score_adv
        if score > max_score and PASS_THRESHOLD_S2 * n <= (n - score_clean):
            max_score = score
            best = i
        scores_clean.append(n - score_clean)
        scores_adv.append(score_adv)
        params.append(i)
        i += increment
    print(*[f'{p:.1f}' for p in params], sep=', ')
    print(*scores_clean, sep=', ')
    print(*scores_adv, sep=', ')
    return best, max_score


def update_one_fold(data_container,
                    model,
                    attack,
                    x_train,
                    y_train,
                    x_eval,
                    y_eval):
    dim = data_container.dim_data
    dc = CustomDataContainer(
        x_train, y_train, x_eval, y_eval,
        name='MNIST_FOLD', data_type='image', num_classes=10, dim_data=dim)
    dc(normalize=True)
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    accuracy = mc.evaluate(x_eval, y_eval)
    print(f'Accuracy on adv. examples: {accuracy}')

    ad = ApplicabilityDomainContainer(
        mc, hidden_model=model.hidden_model, k2=K2,
        reliability=RELIABILITY, sample_ratio=SAMPLE_RATIO,
        kappa=KAPPA, confidence=CONFIDENCE)
    adv, pred_adv, x_clean, y_true = attack.generate(
        use_testset=False, x=x_eval)

    max_scores = np.zeros(2, dtype=np.float32)
    # Search parameters for Stage 2
    zeta = RELIABILITY
    k2 = K2
    # find best k2
    k2, max_score = search_param(
        ad, update_k2, def_stage2, K_RANGE, 1, zeta,
        adv, pred_adv, x_clean, y_true)
    logger.debug('Found best k2: %i with score: %i', k2, max_score)

    # find best zeta
    zeta, max_scores[0] = search_param(
        ad, update_zeta, def_stage2, Z_RANGE, 0.1, k2,
        adv, pred_adv, x_clean, y_true)
    logger.debug('Found best zeta: %f with score: %i', zeta, max_scores[0])

    # Search parameters for Stage 3
    kappa = KAPPA
    gamma = CONFIDENCE
    # find kappa
    # kappa, max_score = search_param(
    #     ad, update_kappa, def_stage3, KAPPA_RANGE, 1, gamma,
    #     adv, pred_adv, x_clean, y_true)
    # logger.debug('Found best kappa: %i with score: %i', kappa, max_score)

    # find parameter gamma
    gamma, max_scores[1] = search_param(
        ad, update_gamma, def_stage3, GAMMA_RANGE, 0.1, kappa,
        adv, pred_adv, x_clean, y_true)
    logger.debug('Found best gamma: %f with score: %i', gamma, max_scores[1])

    return k2, zeta, kappa, gamma, np.sum(max_scores)


def cross_validation(data_container, model, attack):
    start = time.time()
    k2s = np.zeros(NUM_FOLDS, dtype=np.int)
    zetas = np.zeros(NUM_FOLDS, dtype=np.float32)
    kappas = np.zeros(NUM_FOLDS, dtype=np.int)
    gammas = np.zeros(NUM_FOLDS, dtype=np.float32)
    scores = np.zeros(NUM_FOLDS, dtype=np.float32)
    for i in range(NUM_FOLDS):
        start_fold = time.time()
        x_train, y_train, x_eval, y_eval = cross_validation_split(
            data_container.x_train, data_container.y_train, i, NUM_FOLDS)
        k2s[i], zetas[i], kappas[i], gammas[i], scores[i] = update_one_fold(
            data_container, model, attack, x_train, y_train, x_eval, y_eval)
        elapsed_fold = time.time() - start_fold
        print('[{}/{}] {:.0f}m {}s - score: {} k2:{}, zeta:{:.1f}, kappa: {}, gamma: {:.1f}'.format(
            i, NUM_FOLDS, elapsed_fold // 60, elapsed_fold % 60,
            scores[i], k2s[i], zetas[i], kappas[i], gammas[i]))
    max_idx = np.argmax(scores)
    elapsed = time.time() - start
    print('Time to complete cross validation: {:.0f}m {:.3f}s'.format(
        elapsed // 60, elapsed % 60))
    return k2s[max_idx], zetas[max_idx], kappas[max_idx], gammas[max_idx]


def main():
    dc = DataContainer(DATASET_LIST[NAME], get_data_path())
    dc()

    model = MnistCnnV2()
    mc = ModelContainerPT(model, dc)
    mc.load(MODEL_FILE)
    accuracy = mc.evaluate(dc.x_test, dc.y_test)
    print(f'Accuracy on test set: {accuracy}')

    carlini_attack = attacks.CarliniL2V2Container(
        mc,
        learning_rate=0.01,
        binary_search_steps=9,
        max_iter=1000,
        confidence=0.0,
        initial_const=0.01,
        c_range=(0, 1e10),
        batch_size=BATCH_SIZE,
        clip_values=(0.0, 1.0),
    )
    # adv, y_adv, x_clean, y_clean = carlini_attack.generate(count=1000)
    # carlini_attack.save_attack(
    #     'MnistCnnV2_MNIST_Carlini',
    #     adv,
    #     y_adv,
    #     x_clean,
    #     y_clean,
    #     True,
    # )

    # use pre-trained adversarial examples
    adv, y_adv, x_clean, y_clean = carlini_attack.load_adv_examples(
        CARLINI_FILE)

    accuracy = mc.evaluate(adv, y_clean)
    print(f'Accuracy on adv. examples: {accuracy}')

    bim_attack = attacks.BIMContainer(
        mc,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False,
    )
    k2, zeta, kappa, gamma = cross_validation(dc, model, bim_attack)

    # use pre-defined parameters
    # k2, zeta, kappa, gamma, score = K2, RELIABILITY, KAPPA, CONFIDENCE, 0

    ad = ApplicabilityDomainContainer(
        mc,
        hidden_model=model.hidden_model,
        k2=k2,
        reliability=zeta,
        sample_ratio=SAMPLE_RATIO,
        kappa=kappa,
        confidence=gamma,
    )
    print(ad.params)

    ad.fit()
    blocked_indices, x_passed = ad.detect(
        adv, y_adv, return_passed_x=True)
    print('After update parameters, blocked {}/{} samples from adv. examples'.format(
        len(blocked_indices),
        len(adv)))


if __name__ == '__main__':
    master_seed(SEED)
    main()
