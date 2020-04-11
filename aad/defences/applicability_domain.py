"""
This module implements the Applicability Domain based adversarial detection.
"""
import logging

import numpy as np
import sklearn.neighbors as knn
import torch
from scipy import stats
from torch.utils.data import DataLoader

from ..datasets import NumericalDataset
from ..utils import swap_image_channel
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class ApplicabilityDomainContainer(DetectorContainer):
    """
    Class performs adversarial detection based on Applicability Domain.
    """

    def __init__(self,
                 model_container,
                 hidden_model=None,
                 k1=9,
                 reliability=0.8,
                 sample_ratio=1.0,
                 confidence=0.9,
                 kappa=6,
                 disable_s2=False):
        """Create a class `ApplicabilityDomainContainer` instance.

        :param model_container: A trained model
        :type model_container: `ModelContainerPT`
        :param hidden_model: To compute output from certain hidden layer
        :type hidden_model: `torch.nn.Module`
        :param k1: Number of nearest neighbours for Stage 2
        :type k1: `int`
        :param reliability: The parameter for confidence interval in Stage 2
        :type reliability: `float`
        :param sample_ratio: The percentage of train sample will be used in Stage 3. Expected to be in range (0, 1]
        :type sample_ratio: `float`
        :param confidence: The number of samples will be used to compute neighbours in Stage 3. k = num_classes * kappa
        :type confidence: `float`
        :param kappa: The number of samples will be used to compute neighbours in Stage 3. k = num_classes * kappa
        :type kappa: `float`
        :param disable_s2: To disable Stage 2 defence. (For testing the robustness Stage 3)
        :type disable_s2: `bool`
        """
        super(ApplicabilityDomainContainer, self).__init__(model_container)

        params_received = {
            'k1': k1,
            'reliability': reliability,
            'sample_ratio': sample_ratio,
            'confidence': confidence,
            'kappa': kappa,
            'disable_s2': disable_s2,
        }
        self.params = params_received
        self.device = model_container.device
        dc = model_container.data_container
        self.num_classes = dc.num_classes
        self.data_type = dc.data_type

        if hidden_model is not None:
            self.hidden_model = hidden_model
        else:
            self.hidden_model = self.dummy_model

        # placeholders for the objects used by AD
        self.num_components = 0  # number of hidden components
        self.encode_train_np = None
        self.y_train_np = None
        # keep track max for each class, size: [num_classes, num_components]
        self._x_max = None
        self._x_min = None
        self._s2_models = []  # in-class KNN models
        self._s3_model = None  # KNN models using training set
        self._s2_means = np.zeros(self.num_classes, dtype=np.float32)
        self._s2_stds = np.zeros_like(self._s2_means)
        self._s2_thresholds = np.zeros_like(self._s2_means)
        self._s3_likelihood = None

    def fit(self):
        """
        Train the model using the training set from `model_container.data_container`.
        """
        if self.encode_train_np is not None:
            logger.warning(
                'You cannot call fit() method multiple times! Please start a new instance')
            return False

        disable_s2 = self.params['disable_s2']

        self._log_time_start()

        # Step 1: compute hidden layer outputs from inputs
        dc = self.model_container.data_container
        x_train_np = dc.data_train_np
        self.encode_train_np = self._preprocessing(x_train_np)
        self.y_train_np = dc.label_train_np
        self.num_components = self.encode_train_np.shape[1]
        logger.debug('Number of input attributes: %d', self.num_components)

        # Stage 1
        self._fit_stage1()
        self._log_time_end('AD Stage 1')
        # Stage 2
        if disable_s2 is False:
            self._log_time_start()
            self._fit_stage2()
            self._log_time_end('AD Stage 2')
        # Stage 3
        self._log_time_start()
        self._fit_stage3()
        self._log_time_end('AD Stage 3')

        return True
    
    def save(self, filename, overwrite=False):
        """
        The parameters to save for Applicability Domain. Consider save the constants
        in a JSON file.
        """
        pass

    def load(self, filename):
        """
        This method does not support load parameters.
        """
        pass

    def detect(self, adv, pred=None):
        n = len(adv)
        # 1: passed test, 0: blocked by AD
        passed = np.ones(n, dtype=np.int8)

        # The defence does NOT know the true class of adversarial examples. It
        # computes predictions instead.
        if pred_adv is None:
            pred_adv = self.model_container.predict(adv)

        # The adversarial examples exist in image/data space. The KNN model runs
        # in hidden layer (encoded space)
        encoded_adv = self._preprocessing(adv)

        disable_s2 = self.params['disable_s2']

        # Stage 1
        passed = self._def_state1(encoded_adv, pred_adv, passed)
        blocked = len(passed[passed == 0])
        logger.debug('Stage 1: blocked %d inputs', blocked)
        blocked_s1 = blocked
        # Stage 2
        blocked_s2 = 0
        if disable_s2 is False:
            passed = self._def_state2(encoded_adv, pred_adv, passed)
            blocked = len(passed[passed == 0]) - blocked
            logger.debug('Stage 2: blocked %d inputs', blocked)
            blocked_s2 = blocked
        # Stage 3
        passed = self._def_state3_v2(encoded_adv, passed)
        blocked = len(passed[passed == 0]) - blocked
        logger.debug('Stage 3: blocked %d inputs', blocked)
        blocked_s3 = blocked

        passed_indices = np.nonzero(passed)
        blocked_indices = np.delete(np.arange(n), passed_indices)
        return adv[passed_indices], blocked_indices, [blocked_s1, blocked_s2, blocked_s3]

    def _preprocessing(self, x_np):
        # the # of channels should alway smaller than the size of image
        if self.data_type == 'image' and x_np.shape[1] not in (1, 3):
            logger.debug('Before swap channel: x_np: %s', str(x_np.shape))
            x_np = swap_image_channel(x_np)

        dataset = NumericalDataset(torch.as_tensor(x_np))
        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0)

        # run 1 sample to get size of output
        x, _ = next(iter(dataloader))
        x = x.to(self.device)
        outputs = self.hidden_model(x[:1])
        num_components = outputs.size(1)  # number of hidden components

        x_encoded = -999 * torch.ones(len(x_np), num_components)
        start = 0
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                batch_size = len(x)
                x_out = self.hidden_model(x).view(batch_size, -1)  # flatten
                x_encoded[start: start+batch_size] = x_out
                start = start + batch_size
        return x_encoded.cpu().detach().numpy()

    def _fit_stage1(self):
        self._x_min = np.empty(
            (self.num_classes, self.num_components), dtype=np.float32)
        self._x_max = np.empty_like(self._x_min)

        for i in range(self.num_classes):
            indices = np.where(self.y_train_np == i)[0]
            x = self.encode_train_np[indices]
            self._x_max[i] = np.amax(x, axis=0)
            self._x_min[i] = np.amin(x, axis=0)

    def _fit_stage2(self):
        self._s2_models = []
        k1 = self.params['k1']
        zeta = self.params['reliability']
        for i in range(self.num_classes):
            indices = np.where(self.y_train_np == i)[0]
            x = self.encode_train_np[indices]
            # models are grouped by classes, y doesn't matter
            y = np.ones(len(x))
            model = knn.KNeighborsClassifier(n_neighbors=k1, n_jobs=-1)
            model.fit(x, y)
            self._s2_models.append(model)
            # number of neighbours is k + 1, since it will return the node itself
            dist, _ = model.kneighbors(x, n_neighbors=k1+1)
            avg_dist = np.sum(dist, axis=1) / float(k1)
            self._s2_means[i] = np.mean(avg_dist)
            self._s2_stds[i] = np.std(avg_dist)
            self._s2_thresholds[i] = self._s2_means[i] + \
                zeta * self._s2_stds[i]

    def _fit_stage3(self):
        x = self.encode_train_np
        y = self.y_train_np
        kappa = self.params['kappa']
        k = int(self.num_classes * kappa)
        sample_ratio = self.params['sample_ratio']
        logger.debug('k for Stage 3: %d', k)

        self._s3_model = knn.KNeighborsClassifier(
            n_neighbors=k,
            n_jobs=-1,
        )
        self._s3_model.fit(x, y)

        # compute the likelihood
        sample_size = int(np.floor(len(x) * sample_ratio))
        logger.debug('[AD Stage 3]: Size of train set: %d', sample_size)
        x_sub = np.random.permutation(x)[:sample_size]
        neigh_indices = self._s3_model.kneighbors(
            x_sub,
            n_neighbors=k,
            return_distance=False)
        neigh_labels = np.array(
            [self.y_train_np[n_i] for n_i in neigh_indices], dtype=np.int16)
        bins = np.zeros((sample_size, self.num_classes), dtype=np.float32)
        # Is there any vectorization way to compute the histogram?
        for i in range(sample_size):
            bins[i] = stats.relfreq(
                neigh_labels[i],
                numbins=self.num_classes,
                defaultreallimits=(0, self.num_classes-1)
            )[0]
        self._s3_likelihood = np.mean(np.amax(bins, axis=1))
        logger.debug('Train set likelihood = %f', self._s3_likelihood)

    def _def_state1(self, adv, pred_adv, passed):
        """
        A bounding box which uses [min, max] from traning set
        """
        if len(np.where(passed == 1)[0]) == 0:
            return passed

        for i in range(self.num_classes):
            indices = np.where(pred_adv == i)[0]
            x = adv[indices]
            i_min = self._x_min[i]
            i_max = self._x_max[i]
            blocked_indices = np.where(
                np.all(np.logical_or(x < i_min, x > i_max), axis=1)
            )[0]
            if len(blocked_indices) > 0:
                passed[blocked_indices] = 0
        return passed

    def _def_state2(self, adv, pred_adv, passed):
        """
        Filtering the inputs based on in-class k nearest neighbours.
        """
        passed_indices = np.where(passed == 1)[0]
        if len(passed_indices) == 0:
            return passed

        indices = np.arange(len(adv))
        passed_adv = adv[passed_indices]
        passed_pred = pred_adv[passed_indices]
        models = self._s2_models
        classes = np.arange(self.num_classes)
        k1 = self.params['k1']

        for model, threshold, c in zip(models, self._s2_thresholds, classes):
            inclass_indices = np.where(passed_pred == c)[0]
            if len(inclass_indices) == 0:
                continue

            x = passed_adv[inclass_indices]
            neigh_dist, neigh_indices = model.kneighbors(
                x, n_neighbors=k1, return_distance=True)
            mean = np.mean(neigh_dist, axis=1)
            sub_blocked_indices = np.where(mean > threshold)[0]
            # trace back the original indices from input adversarial examples
            blocked_indices = indices[passed_indices][inclass_indices][sub_blocked_indices]

            if len(blocked_indices) > 0:
                passed[blocked_indices] = 0

        return passed

    def _def_state3(self, adv, pred_adv, passed):
        """
        NOTE: Deprecated!
        Filtering the inputs based on k nearest neighbours with entire training set
        """
        passed_indices = np.where(passed == 1)[0]
        if len(passed_indices) == 0:
            return passed

        indices = np.arange(len(adv))
        passed_adv = adv[passed_indices]
        passed_pred = pred_adv[passed_indices]
        model = self._s3_model
        knn_pred = model.predict(passed_adv)
        not_match_indices = np.where(np.not_equal(knn_pred, passed_pred))[0]
        blocked_indices = indices[passed_indices][not_match_indices]

        if len(blocked_indices) > 0:
            passed[blocked_indices] = 0

        return passed

    def _def_state3_v2(self, adv, passed):
        """
        Checking the class distribution of k nearest neighbours without predicting
        the inputs. Compute the likelihood using one-against-all approach. 
        """
        passed_indices = np.where(passed == 1)[0]
        if len(passed_indices) == 0:
            return passed

        x = adv[passed_indices]
        kappa = self.params['kappa']
        k = self.num_classes * kappa
        confidence = self.params['confidence']

        model = self._s3_model  # KNeighborsClassifier for entire train set
        neigh_indices = model.kneighbors(
            x, n_neighbors=k, return_distance=False)
        neigh_labels = np.array(
            [self.y_train_np[n_i] for n_i in neigh_indices], dtype=np.int16)
        bins = np.zeros((len(x), self.num_classes), dtype=np.float32)

        for i in range(len(x)):
            bins[i] = stats.relfreq(
                neigh_labels[i],
                numbins=self.num_classes,
                defaultreallimits=(0, self.num_classes-1)
            )[0]

        likelihood = np.amax(bins, axis=1)
        logger.debug('Mean likelihood on adv: %f', likelihood.mean())
        threshold = self._s3_likelihood * confidence
        blocked_indices = np.where(likelihood < threshold)[0]
        passed[blocked_indices] = 0

        return passed

    @staticmethod
    def dummy_model(inputs):
        """
        Return the input.Use this method when we don't need a hidden layer encoding.
        """
        return inputs
