"""
This module implements the Feature Squeezing defence.
"""
import copy
import logging
import os

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.stats import mode
from sklearn.tree import ExtraTreeClassifier

from ..basemodels import ModelContainerTree, copy_model
from ..utils import name_handler, scale_normalize
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class FeatureSqueezingTree(DetectorContainer):
    """
    Class preforms Feature Squeezing defence
    """

    def __init__(self,
                 model_container,
                 smoothing_methods,
                 clip_values=None,
                 bit_depth=None,
                 sigma=None,
                 kernel_size=None,
                 pretrained=True):
        """
        Create a FeatureSqueezing class instance.

        Parameters
        ----------
        model_container : ModelContainerTree
            Pre-trained classification model.
        smoothing_methods : list of ('median', 'normal', 'binary')
            Select one or more smoothing methods. 'median' filter can only use on images. 'binary' filter maps features
            based on `bit_depth`. 'normal' filter generates noise based on normal distribution.
        clip_values : tuple of (min, max)
            Indicates the minimum and maximum values of each feature. If the data is normalized, it should be (0.0, 1.0).
        bit_depth : int, optional
            The image color depth for input images. Pass `None` for numeral data. Required for 'binary' filter. e.g.: 8
        sigma : float, optional
            The Standard Deviation of normal distribution. Required for 'normal' filter. e.g.: 0.1
        kernel_size: int, optional
            The kernel size for median filter. Required for 'median' filter. e.g.: 3
        pretrained : bool
            Load the pre-trained parameters before train the smoothing models.
        """
        super(FeatureSqueezingTree, self).__init__(model_container)

        if not isinstance(model_container, ModelContainerTree):
            raise ValueError(
                'FeatureSqueezingTree only supports Tree Model. Received {}'.format(
                    type(model_container)))

        data_container = model_container.data_container
        if clip_values is None:
            clip_values = data_container.data_range

        self._params = {
            'clip_values': clip_values,
            'bit_depth': bit_depth,
            'sigma': sigma,
            'kernel_size': kernel_size,
            'pretrained': pretrained,
        }
        self._smoothing_methods = smoothing_methods
        if 'median' in smoothing_methods \
                and data_container.data_type is not 'image':
            raise ValueError('median filter is only avaliable for images.')

        if 'median' in smoothing_methods and kernel_size is None:
            raise ValueError('kernel_size is required.')
        if 'binary' in smoothing_methods and bit_depth is None:
            raise ValueError('bit_depth is required.')
        if 'normal' in smoothing_methods and sigma is None:
            raise ValueError('sigma is required.')

        num_features = data_container.dim_data[0]
        num_classes = data_container.num_classes
        self._models = []
        for method_name in smoothing_methods:
            logger.info('Creating ExtraTreeClassifier for %s...', method_name)
            classifier = ExtraTreeClassifier(
                criterion='gini',
                splitter='random',
            )
            mc = ModelContainerTree(classifier, copy.deepcopy(data_container))
            dc = mc.data_container

            # replace train set with squeezed dataset
            if method_name == 'binary':
                x_train = self.apply_binary_transform(dc.x_train)
                x_test = self.apply_binary_transform(dc.x_test)
            elif method_name == 'normal':
                x_train = self.apply_normal_transform(dc.x_train)
                x_test = self.apply_normal_transform(dc.x_test)
            elif method_name == 'median':
                x_train = self.apply_median_transform(dc.x_train)
                x_test = self.apply_median_transform(dc.x_test)
            mc.data_container.x_train = x_train
            mc.data_container.y_train = np.copy(dc.y_train)
            mc.data_container.x_test = x_test
            mc.data_container.y_test = np.copy(dc.y_test)
            self._models.append({
                'name': method_name,
                'model_container': mc,
            })

    @property
    def smoothing_methods(self):
        """Get the name list of the squeezing methods"""
        return self._smoothing_methods

    def fit(self):
        """
        Train the smoothing models with selected filters. Since each model uses seperated train set, we will train it
        seperately. All models are using the same test set.

        Parameters
        ----------
        max_epochs : int
            Number of epochs the program will run during the training.
        batch_size : int
            Size of a mini-batch.
        """
        for model in self._models:
            self._log_time_start()
            name = model['name']
            logger.debug('Start training %s squeezing model...', name)
            mc = model['model_container']
            mc.fit()
            self._log_time_end(f'Train {name}')

    def save(self, filename, overwrite=False):
        """Save trained parameters."""
        logger.warning('FeatureSqueezingTree does NOT support save')
        raise NotImplementedError(
            'Feature Squeezing for Tree Classifier does NOT support save')

    def load(self, filename):
        """Load pre-trained parameters."""
        logger.warning('FeatureSqueezingTree does NOT support load')
        raise NotImplementedError(
            'Feature Squeezing for Tree Classifier does NOT support load')

    def does_pretrained_exist(self, filename):
        """Do pretrained files exist?"""
        idx = filename.find('.pt')
        filename = filename[:idx] if idx != -1 else filename
        for model in self._models:
            name = model['name']
            method_filename = name_handler(filename + '_' + name, 'pt', True)
            if not os.path.exists(method_filename):
                return False
        return True

    def detect(self, adv, pred=None, return_passed_x=False):
        """
        Compare the predictions between squeezed models and initial model. Block any mismatches.

        Parameters
        ----------
        adv : numpy.ndarray
            The data for evaluation.
        pred : numpy.ndarray, optional
            The predictions of the input data. If it is none, this method will use internal model to make prediction.
        return_passed_x : bool
            The flag of returning the data which are passed the test.

        Returns
        -------
        blocked_indices : numpy.ndarray
            List of blocked indices.
        x_passed : numpy.ndarray
            The data which are passed the test. This parameter will not be returns if `return_passed_x` is False.
        """
        # use -1 as a place holder
        results = -1 * np.ones((len(self._models) + 1,
                                len(adv)), dtype=np.int64)

        mc = self.model_container
        my_pred = mc.predict(adv)
        if pred is None:
            pred = my_pred
        assert np.all(pred == my_pred)
        results[0] = pred

        i = 1
        for model in self._models:
            method_name = model['name']
            mc = model['model_container']
            if method_name == 'binary':
                adv_trans = self.apply_binary_transform(adv)
            elif method_name == 'normal':
                adv_trans = self.apply_normal_transform(adv)
            elif method_name == 'median':
                adv_trans = self.apply_median_transform(adv)
            results[i] = mc.predict(adv_trans)
            i += 1

        # they have same labels, if variance is 0.
        matched = np.std(results, axis=0) == 0
        passed_indices = np.where(matched)[0]
        blocked_indices = np.where(np.logical_not(matched))[0]
        if return_passed_x:
            # return the data without transformation
            return blocked_indices, adv[passed_indices]
        return blocked_indices

    def evaluate(self, x, labels):
        """
        Given a list of samples, evaluate the accuracy of the classification model.

        Parameters
        ----------
        x : numpy.ndarray, torch.Tensor
            Input data for evaluation.
        labels : numpy.ndarray, torch.Tensor
            The true labels of x.

        Returns
        -------
        accuracy : float
            The accuracy of the predictions.
        """
        if len(x) == 0:
            return 0.0

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        # use -1 as a place holder
        results = -1 * np.ones((len(self._models) + 1,
                                len(x)), dtype=np.int64)

        mc = self.model_container
        results[0] = mc.predict(x)

        i = 1
        for model in self._models:
            method_name = model['name']
            mc = model['model_container']
            if method_name == 'binary':
                adv_trans = self.apply_binary_transform(x)
            elif method_name == 'normal':
                adv_trans = self.apply_normal_transform(x)
            elif method_name == 'median':
                adv_trans = self.apply_median_transform(x)
            results[i] = mc.predict(adv_trans)
            i += 1

        pred = mode(results, axis=0)[0].reshape(labels.shape)
        accuracy = np.sum(np.equal(pred, labels)) / len(labels)
        return accuracy

    def get_def_model_container(self, method):
        """
        Get the squeezing model container

        Parameters
        ----------
        method : str
            The name of the squeezing method.

        Returns
        -------
        ModelContainer
            The selected model container.
        """
        for model in self._models:
            if model['name'] == method:
                return model['model_container']

    def apply_binary_transform(self, x):
        """
        Apply binary transformation on input x. Rescale the input based on given bit depth. The parameters for
        transformation were predefined when creating class instance.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        x : np.ndarray
            Results after binary transformation.
        """
        clip_values = self._params['clip_values']
        bit_depth = self._params['bit_depth']

        x_norm = scale_normalize(x, clip_values[0], clip_values[1])
        max_val = np.rint(2 ** bit_depth - 1)
        res = np.rint(x_norm * max_val) / max_val
        res = res * (clip_values[1] - clip_values[0])
        res += clip_values[0]
        return res.astype(np.float32)

    def apply_normal_transform(self, x):
        """
        Add noise with Normal distribution to input x. The parameters for transformation were predefined when creating
        class instance.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        x : np.ndarray
            Input parameter x with noise.
        """
        sigma = self._params['sigma']
        clip_values = self._params['clip_values']
        shape = x.shape

        res = x + np.random.normal(0, scale=sigma, size=shape)
        res = np.clip(res, clip_values[0], clip_values[1])
        return res.astype(np.float32)

    def apply_median_transform(self, x):
        """
        Apply median filter on a given input x. The parameters for transformation were predefined when creating class
        instance.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        x : np.ndarray
            Input parameter x after median filer.
        """
        kernel_size = self._params['kernel_size']

        res = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            res[i] = median_filter(x[i], size=kernel_size)
        return res.astype(np.float32)
