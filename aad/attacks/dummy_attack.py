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
        n = len(self.dc.data_test_np)
        if count is not 'all':
            shuffled_indices = np.random.permutation(n)[:count]
            x = self.dc.data_test_np[shuffled_indices]
            y = self.dc.label_test_np[shuffled_indices]
        else:
            x = self.dc.data_test_np
            y = self.dc.label_test_np
        pred = self.mc.predict(x)
        return x, pred, x, y
