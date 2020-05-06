"""
This module implement the base class for scikit-learn tree model container.
"""
import logging
import os
import time

import numpy as np
import sklearn.tree as tree

from ..datasets import DataContainer

logger = logging.getLogger(__name__)


class ModelContainerTree:
    """
    This class provides additional features for the sklearn tree classifier.
    """

    def __init__(self, model, data_container):
        """
        Create a ModelContainerTree class instance

        Parameters
        ----------
        model : DecisionTreeClassifier or ExtraTreeClassifier
            A sklearn tree classifier
        data_container : DataContainer
            An instance of DataContainer
        """
        assert isinstance(model, tree.BaseDecisionTree), \
            f'Expecting a sklearn.tree classifier, got {type(model)}'
        self._model = model
        assert isinstance(data_container, DataContainer), \
            f'Expecting a DataContainer, got {type(data_container)}'
        self.data_container = data_container

        # It's a CPU only implementation
        self.device = 'cpu'

    @property
    def model(self):
        """Get the sklearn tree classifier model."""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def fit(self):
        """
        Train the classifier
        """
        since = time.time()
        x = self.data_container.x_train
        y = self.data_container.y_train
        self._model.fit(x, y)

        time_elapsed = time.time() - since
        logger.info('Time to complete training: %dm %.3fs',
                    int(time_elapsed // 60), time_elapsed % 60)

    @staticmethod
    def hidden_model(x):
        """
        A dummy method. No hidden model for tree classifier.
        """
        return x

    def predict(self, x):
        """
        Predicts a list of samples.

        Parameters
        ----------
        x : numpy.ndarray, torch.Tensor
            Input data for forward propagation.

        Returns
        -------
        predictions : numpy.ndarray
            The predicted labels.
        """
        if len(x) == 0:
            return np.array([], dtype=np.int64)
        return self._model.predict(x)

    def predict_one(self, x):
        """
        Predicts single input.

        Parameters
        ----------
        x : numpy.ndarray, torch.Tensor
            An input sample.

        Returns
        -------
        prediction : int
            The predicted label.
        """
        if len(x) == 0:
            return np.array([], dtype=np.int64)
        x = np.expand_dims(x, axis=0)
        prediction = self.predict(x)
        return prediction.squeeze()

    def evaluate(self, x, labels):
        """
        Given a list of samples, evaluate the accuracy of the classification model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data for evaluation.
        labels : numpy.ndarray
            The true labels of x.

        Returns
        -------
        accuracy : float
            The accuracy of the predictions.
        """
        if len(x) == 0:
            return 0.0

        predictions = self.predict(x)
        accuracy = np.sum(np.equal(predictions, labels)) / len(labels)
        return accuracy
