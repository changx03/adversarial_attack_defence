from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from aad.utils import master_seed, get_range

logger = logging.getLogger(__name__)

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

        logger.info('Passed test 1')


if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.INFO)
    unittest.main()