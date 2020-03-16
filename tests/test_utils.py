import logging
import os
import unittest

import numpy as np

from aad.utils import (get_range, master_seed, name_handler, onehot_encoding,
                       scale_normalize, scale_unnormalize, shuffle_data,
                       swap_image_channel)

logger = logging.getLogger(os.path.basename(__file__))
SEED = 4096


class TestUtils(unittest.TestCase):
    def setUp(self):
        master_seed(SEED)

    def test_master_seed_np(self):
        master_seed(SEED)
        a = np.random.rand(10)
        b = np.random.rand(10)

        master_seed(SEED)
        z = np.random.rand(10)

        self.assertTrue((a != b).any())
        self.assertTrue((z == a).all())

    def test_master_seed_torch(self):
        import torch

        master_seed(SEED)
        a = torch.randn(10)
        b = torch.randn(10)

        master_seed(SEED)
        z = torch.randn((10))

        self.assertTrue((a != b).any())
        self.assertTrue((z == a).all())

    def test_get_range(self):
        x = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.float32)
        out = get_range(x)
        e = ([0., 0.], [2., 2.])
        r = [all(xx == ee) for xx, ee in zip(out, e)]
        self.assertTrue(all(r))

        master_seed(SEED)
        x = np.random.rand(10, 1, 28, 28)
        out = np.round(np.array(list(get_range(x, is_image=True))) * 1e4)
        self.assertTrue((out == [0., 9999.]).all())

    def test_normalize(self):
        master_seed(SEED)
        x = np.random.rand(2, 3)
        xmin, xmax = get_range(x)
        x_norm = scale_normalize(x, xmin, xmax)
        x_unnorm = scale_unnormalize(x_norm, xmin, xmax)
        self.assertTrue(np.array_equiv(
            np.round(x*1e4), np.round(x_unnorm*1e4)))

    def test_shuffle_data(self):
        master_seed(SEED)
        x = np.random.rand(3)
        out = shuffle_data(x)
        self.assertTrue(x.shape == out.shape)
        self.assertTrue(all([xx in x for xx in out]))
        self.assertFalse((x == out).all())

        from pandas import DataFrame
        d = DataFrame(x, columns=['value'], dtype=np.float32)
        out = shuffle_data(d)
        self.assertTrue(d.shape == out.shape)
        self.assertTrue(all([xx in d.values for xx in out.values]))
        self.assertFalse((d.values == out.values).all())

    def test_swap_image_channel(self):
        master_seed(SEED)
        x = np.random.rand(1, 10, 10)
        out = swap_image_channel(x)
        self.assertTrue(out.shape == (10, 10, 1))

        x = np.random.rand(1, 10, 10, 3)
        out = swap_image_channel(x)
        self.assertTrue(out.shape == (1, 3, 10, 10))

    def test_name_handler(self):
        x = name_handler('test', 'test', overwrite=True)
        self.assertEqual(x, 'test.test')

    def test_onehot_encoding(self):
        x = np.array([0, 1])
        out = onehot_encoding(x, 3)
        e = [[1, 0, 0], [0, 1, 0]]
        self.assertTrue((out == e).all())


if __name__ == '__main__':
    unittest.main()
