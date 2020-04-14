"""
This module implements the Feature Squeezing defence.
"""
import copy
import logging

import numpy as np
from scipy.ndimage import median_filter

from ..basemodels import ModelContainerPT
from ..utils import copy_model, name_handler, scale_normalize
from .detector_container import DetectorContainer

logger = logging.getLogger(__name__)


class FeatureSqueezing(DetectorContainer):
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
        model_container : ModelContainerPT
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
        super(FeatureSqueezing, self).__init__(model_container)

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

        self._models = []
        for method_name in smoothing_methods:
            model = copy_model(model_container.model, pretrained)
            mc = ModelContainerPT(model, copy.deepcopy(data_container))
            dc = mc.data_container

            # replace train set with squeezed dataset
            if method_name == 'binary':
                x_train, y_train = self.apply_binary_transform(
                    dc.x_train, dc.y_train)
                x_test, y_test = self.apply_binary_transform(
                    dc.x_test, dc.y_test)
            elif method_name == 'normal':
                x_train, y_train = self.apply_normal_transform(
                    dc.x_train, dc.y_train)
                x_test, y_test = self.apply_normal_transform(
                    dc.x_test, dc.y_test)
            elif method_name == 'median':
                x_train, y_train = self.apply_median_transform(
                    dc.x_train, dc.y_train)
                x_test, y_test = self.apply_median_transform(
                    dc.x_test, dc.y_test)
            mc.data_container.x_train = x_train
            mc.data_container.y_train = y_train
            mc.data_container.x_test = x_test
            mc.data_container.y_test = y_test
            self._models.append({
                'name': method_name,
                'model_container': mc,
            })

    @property
    def smoothing_methods(self):
        """Get the name list of the squeezing methods"""
        return self._smoothing_methods

    def fit(self, max_epochs=10, batch_size=128):
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
            mc.fit(max_epochs, batch_size, early_stop=False)
            self._log_time_end(f'Train {name}')

    def save(self, filename, overwrite=False):
        """Save trained parameters."""
        for model in self._models:
            name = model['name']
            method_filename = name_handler(filename + '_' + name, 'pt', True)
            model['model_container'].save(method_filename, overwrite)

    def load(self, filename):
        """Load pre-trained parameters."""
        for model in self._models:
            name = model['name']
            method_filename = name_handler(filename + '_' + name, 'pt', True)
            model['model_container'].load(method_filename)

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
                adv_trans, _ = self.apply_binary_transform(adv, pred)
            elif method_name == 'normal':
                adv_trans, _ = self.apply_normal_transform(adv, pred)
            elif method_name == 'median':
                adv_trans, _ = self.apply_median_transform(adv, pred)
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

    def apply_binary_transform(self, x, y):
        """
        Apply binary transformation on input x. Rescale the input based on given bit depth. The parameters for
        transformation were predefined when creating class instance.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        y : np.ndarray
            The labels for the data.
        Returns
        -------
        x : np.ndarray
            Results after binary transformation.

        y : np.ndarray
            A copy of input parameter y.
        """
        clip_values = self._params['clip_values']
        bit_depth = self._params['bit_depth']

        x_norm = scale_normalize(x, clip_values[0], clip_values[1])
        max_val = np.rint(2 ** bit_depth - 1)
        res = np.rint(x_norm * max_val) / max_val
        res = res * (clip_values[1] - clip_values[0])
        res += clip_values[0]
        return res.astype(np.float32), np.copy(y)

    def apply_normal_transform(self, x, y):
        """
        Add noise with Normal distribution to input x. The parameters for transformation were predefined when creating
        class instance.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        y : np.ndarray
            The labels for the data.
        Returns
        -------
        x : np.ndarray
            Input parameter x with noise.

        y : np.ndarray
            A copy of input parameter y.
        """
        sigma = self._params['sigma']
        clip_values = self._params['clip_values']
        shape = x.shape

        res = x + np.random.normal(0, scale=sigma, size=shape)
        res = np.clip(res, clip_values[0], clip_values[1])
        return res.astype(np.float32), np.copy(y)

    def apply_median_transform(self, x, y):
        """
        Apply median filter on a given input x. The parameters for transformation were predefined when creating class
        instance.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        y : np.ndarray
            The labels for the data.
        Returns
        -------
        x : np.ndarray
            Input parameter x after median filer.

        y : np.ndarray
            A copy of input parameter y.
        """
        kernel_size = self._params['kernel_size']

        res = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            res[i] = median_filter(x[i], size=kernel_size)
        return res.astype(np.float32), np.copy(y)
