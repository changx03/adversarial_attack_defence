import numpy as np


class DummyAttack:
    """
    Do nothing. This returns test set from a DataContainer.
    """

    def __init__(self, model_container, shuffle=True):
        self.mc = model_container
        self.dc = model_container.data_container
        self.shuffle = shuffle

    def generate(self, count='all'):
        n = len(self.dc.x_test)
        if count is not 'all':
            shuffled_indices = np.random.permutation(n)[:count]
            x = self.dc.x_test[shuffled_indices]
            y = self.dc.y_test[shuffled_indices]
        else:
            x = self.dc.x_test
            y = self.dc.y_test
        pred = self.mc.predict(x)
        return x, pred, x, y
