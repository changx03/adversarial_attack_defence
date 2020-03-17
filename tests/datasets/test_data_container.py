import logging
import unittest
import os

import numpy as np

from aad.datasets import DataContainer, get_sample_mean, get_sample_std, DATASET_LIST
from aad.utils import master_seed, swap_image_channel

SEED = 4096


class TestDataContainer(unittest.TestCase):
    @staticmethod
    def get_data_path():
        path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, 'data')
        return path

    def setUp(self):
        master_seed(SEED)

    def test_get_sample_mean(self):
        x = get_sample_mean('MNIST')
        self.assertEqual(len(x), 1)

        x = get_sample_mean('TEST_NO_DATA')
        self.assertEqual(x, [0.])

    def test_get_sample_std(self):
        x = get_sample_std('MNIST')
        self.assertEqual(len(x), 1)

        x = get_sample_std('TEST_NO_DATA')
        self.assertEqual(x, [1.])

    def test_file_exist(self):
        self.assertTrue(os.path.exists(self.get_data_path()))

    @classmethod
    def init_datacontainer(cls, name):
        x = DATASET_LIST[name]
        path = cls.get_data_path()
        dc = DataContainer(x, path)
        dc(shuffle=False, normalize=True, size_train=0.5,
            enable_cross_validation=False)
        return dc

    def test_DataContainer_MNIST(self):
        dc = self.init_datacontainer('MNIST')

        # basic props
        r = dc.data_range
        self.assertEqual(r, (0., 1.))
        name = dc.name
        self.assertEqual(name, 'MNIST')
        m = dc.num_classes
        self.assertEqual(m, 10)
        dtype = dc.type
        self.assertEqual(dtype, 'image')
        dim = dc.dim_data
        self.assertEqual(dim, (1, 28, 28))
        mu = dc.train_mean
        self.assertEqual(mu, [0.13066046])
        std = dc.train_std
        self.assertEqual(std, [0.30150425])

        # train set
        train_np = dc.data_train_np
        self.assertEqual(train_np.shape, (60000, 28, 28, 1))
        y_np = dc.label_train_np
        self.assertEqual(y_np.shape, (60000, ))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), 60000)
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), (8, 1, 28, 28))
        self.assertEqual(y_pt.size(), (8, ))
        x1 = train_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.data_test_np
        self.assertEqual(test_np.shape, (10000, 28, 28, 1))
        y_np = dc.label_test_np
        self.assertEqual(y_np.shape, (10000, ))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), 10000)
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), (8, 1, 28, 28))
        self.assertEqual(y_pt.size(), (8, ))
        x1 = test_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)


if __name__ == '__main__':
    unittest.main()
