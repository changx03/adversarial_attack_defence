import logging
import os
import time
import copy

import numpy as np

from aad.datasets import CustomDataContainer
from aad.utils import cross_validation_split, name_handler

logger = logging.getLogger('CV')


class CrossValidation:
    def __init__(self,
                 applicability_domain,
                 num_folds,
                 k_range,
                 z_range,
                 kappa_range,
                 gamma_range,
                 epsilon):
        self.applicability_domain = applicability_domain
        self.model_container = applicability_domain.model_container
        self.num_folds = num_folds
        self.k_range = k_range
        self.z_range = z_range
        self.kappa_range = kappa_range
        self.gamma_range = gamma_range
        self.epsilon = epsilon

        # the optimal parameters
        self.sample_ratio = None
        self.k2 = None
        self.zeta = None
        self.kappa = None
        self.gamma = None

        # results per fold
        self.folds = []
        self.k2s = []
        self.zetas = []
        self.kappas = []
        self.gammas = []
        self.scores = []
        self.blk_cleans = []
        self.blk_advs = []

        self.attack = None
        self.adv = None
        self.pred_adv = None
        self.x_clean = None
        self.y_true = None

    def fit(self, attack):
        params = self.applicability_domain.params
        self.sample_ratio = ['sample_ratio']
        self.k2 = params['k2']
        self.zeta = params['reliability']
        self.kappa = params['kappa']
        self.gamma = params['confidence']
        self.attack = attack
        self.k2, self.zeta, self.kappa, self.gamma = self._cross_validation()

    def save(self, filename):
        filename = name_handler(os.path.join(
            'save', filename), 'csv', overwrite=False)
        title = ['n_fold', 'k2', 'zeta', 'kappa',
                 'gamma', 'score', 'blk_cleans', 'blk_advs']
        with open(filename, 'w') as file:
            line = ','.join(title)
            file.write(line + '\n')
            for i in range(len(self.folds)):
                line_builder = []
                line_builder.append(self.folds[i])
                line_builder.append(self.k2s[i])
                line_builder.append(self.zetas[i])
                line_builder.append(self.kappas[i])
                line_builder.append(self.gammas[i])
                line_builder.append(self.scores[i])
                line_builder.append(self.blk_cleans[i])
                line_builder.append(self.blk_advs[i])
                line = ','.join([str(i) for i in line_builder])
                file.write(line + '\n')
            file.close()

    def _save_k2(self, new_var, params):
        params = copy.deepcopy(params)
        params['k2'] = int(new_var)
        return params

    def _save_zeta(self, new_var, params):
        params = copy.deepcopy(params)
        params['reliability'] = np.around(new_var, decimals=1)
        return params

    def _save_kappa(self, new_var, params):
        params = copy.deepcopy(params)
        params['kappa'] = int(new_var)
        return params

    def _save_gamma(self, new_var, params):
        params = copy.deepcopy(params)
        params['confidence'] = np.around(new_var, decimals=1)
        return params

    def _save_params(self, save_updated_param, new_var, fold, params,
                     score, blk_clean, blk_adv):
        params = save_updated_param(new_var, params)
        self.folds.append(fold)
        self.k2s.append(int(params['k2']))
        self.zetas.append(np.around(params['reliability'], decimals=1))
        self.kappas.append(int(params['kappa']))
        self.gammas.append(np.around(params['confidence'], decimals=1))
        self.scores.append(np.around(score, decimals=1))
        self.blk_cleans.append(blk_clean)
        self.blk_advs.append(blk_adv)

    def _update_k2(self, update, const):
        logger.debug('Received k2: %f, zeta: %f', update, const)
        kwargs = {'k2': int(update)}
        self.applicability_domain.set_params(**kwargs)
        self.applicability_domain.fit_stage2_(update, const)

    def _update_zeta(self, update, const):
        logger.debug('Received k2: %f, zeta: %f', const, update)
        kwargs = {'reliability': np.around(update, decimals=1)}
        self.applicability_domain.set_params(**kwargs)
        self.applicability_domain.fit_stage2_(const, update)

    def _def_stage2(self, encoded_data, labels, passed):
        return self.applicability_domain.def_stage2_(encoded_data, labels, passed)

    def _update_kappa(self, update, const):
        logger.debug('Received kappa: %f, gamma: %f', update, const)
        kwargs = {'kappa': int(update)}
        self.applicability_domain.set_params(**kwargs)
        self.applicability_domain.fit_stage3_(update, const)

    def _update_gamma(self, update, const):
        logger.debug('Received kappa: %f, gamma: %f', const, update)
        kwargs = {'confidence': np.around(update, decimals=1)}
        self.applicability_domain.set_params(**kwargs)
        self.applicability_domain.fit_stage3_(const, update)

    def _def_stage3(self, encoded_data, labels, passed):
        return self.applicability_domain.def_stage3_(encoded_data, labels, passed)

    def _search_param(self, nth_fold, update_fn, def_stage, save_updated_param,
                      drange, increment, const_param):
        if nth_fold == 1:
            print('break')
        ad = self.applicability_domain
        epsilon = self.epsilon
        i = drange[0]
        scores_clean = []  # number of passed clean
        scores_adv = []  # number of blocked adv. examples
        scores = []  # total score
        values = []
        n = len(self.adv)
        while i <= drange[1]:
            update_fn(i, const_param)
            passed = np.ones(n, dtype=np.int8)
            encoded_clean = ad.preprocessing_(self.x_clean)
            passed_clean = def_stage(encoded_clean, self.y_true, passed)
            score_clean = len(passed_clean[passed_clean == 1])
            encoded_adv = ad.preprocessing_(self.adv)
            passed_adv = def_stage(encoded_adv, self.pred_adv, passed)
            score_adv = len(passed_clean[passed_adv == 0])
            scores_clean.append(score_clean)
            scores_adv.append(score_adv)
            score = score_clean - n + epsilon * score_adv
            scores.append(score)
            values.append(i)
            self._save_params(save_updated_param, i, nth_fold,
                              ad.params, score, score_clean, score_adv)
            i += increment
        logger.debug('var: %s', ','.join([str(v) for v in values]))
        logger.debug('score_clean: %s', ','.join(
            [str(v) for v in scores_clean]))
        logger.debug('score_adv: %s', ','.join([str(v) for v in scores_adv]))
        logger.debug('score_total: %s', ','.join([str(v) for v in scores]))

        idx = np.argmax(scores)
        return values[idx], scores[idx]

    def _update_one_fold(self, nth_fold, attack, x_train, y_train, x_eval, y_eval):
        mc = self.model_container
        dim = mc.data_container.dim_data
        num_classes = mc.data_container.num_classes
        name = 'FOLD_' + str(nth_fold)
        dc = CustomDataContainer(
            x_train, y_train, x_eval, y_eval,
            name=name, data_type=mc.data_container.data_type,
            num_classes=num_classes, dim_data=dim)
        dc(normalize=True)
        self.adv, self.pred_adv, self.x_clean, self.y_true = attack.generate(
            use_testset=False, x=x_eval)
        accuracy = mc.evaluate(self.adv, self.y_true)
        logger.info('Accuracy of adv. examples on %s: %f', name, accuracy)

        max_scores = np.zeros(2, dtype=np.float32)
        # Search parameters for Stage 2
        zeta = self.zeta
        k2 = self.k2
        # find best k2
        new_k2, max_score = self._search_param(
            nth_fold, self._update_k2, self._def_stage2, self._save_k2, self.k_range, 1, zeta)
        k2 = int(new_k2)
        logger.debug('Found best k2: %i with score: %i', k2, max_score)

        # find best zeta
        new_zeta, max_score = self._search_param(
            nth_fold, self._update_zeta, self._def_stage2, self._save_zeta, self.z_range, 0.1, k2)
        zeta = np.around(new_zeta, decimals=1)
        logger.debug('Found best zeta: %f with score: %i', zeta, max_scores[0])

        # Search parameters for Stage 3
        kappa = self.kappa
        gamma = self.gamma
        # find kappa
        new_kappa, max_score = self._search_param(
            nth_fold, self._update_kappa, self._def_stage3, self._save_kappa, self.kappa_range, 1, gamma)
        kappa = int(new_kappa)
        logger.debug('Found best kappa: %i with score: %i', kappa, max_score)

        # find parameter gamma
        new_gamma, max_scores[1] = self._search_param(
            nth_fold, self._update_gamma, self._def_stage3, self._save_gamma, self.gamma_range, 0.1, kappa)
        gamma = np.around(new_gamma, decimals=1)
        logger.debug('Found best gamma: %f with score: %i',
                     gamma, max_scores[1])

        return k2, zeta, kappa, gamma, np.sum(max_scores)

    def _cross_validation(self):
        attack = self.attack
        dc = self.model_container.data_container
        num_folds = self.num_folds

        start = time.time()
        for i in range(num_folds):
            start_fold = time.time()
            x_train, y_train, x_eval, y_eval = cross_validation_split(
                dc.x_all, dc.y_all, i, num_folds)
            k2, zeta, kappa, gamma, max_score = self._update_one_fold(
                i, attack, x_train, y_train, x_eval, y_eval)
            elapsed_fold = time.time() - start_fold
            logger.debug('[%d/%d] %.0fm %.1fs - score: %.1f - k2: %d, zeta: %.1f, kappa: %d, gamma: %.1f',
                         i, num_folds, elapsed_fold // 60, elapsed_fold % 60,
                         max_score, k2, zeta, kappa, gamma)
        max_idx = np.argmax(self.scores)
        elapsed = time.time() - start
        logger.info('Time to complete cross validation: %.0fm %.1fs',
                    elapsed // 60, elapsed % 60)
        return self.k2s[max_idx], self.zetas[max_idx], self.kappas[max_idx], self.gammas[max_idx]
