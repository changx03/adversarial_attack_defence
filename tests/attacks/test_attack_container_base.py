import logging
import os
import unittest

import numpy as np

from aad.attacks import AttackContainer
from aad.utils import master_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096  # 2**12 = 4096


class TestAttackContainer(unittest.TestCase):
    """Test AttackContainer class"""

    def setUp(self):
        master_seed(SEED)

    def test_save(self):
        adv = np.random.rand(1000, 32, 32, 3).astype(np.float32)
        pred = np.random.choice(range(10), 1000, replace=True).astype(np.int64)
        x = np.random.rand(1000, 32, 32, 3).astype(np.float32)
        y = np.random.choice(range(10), 1000, replace=True).astype(np.int64)
        file_path = os.path.join('test', 'test_attack')
        AttackContainer.save_attack(file_path, adv, pred, x, y, overwrite=True)

        file_path = os.path.join('save', file_path)
        fadv = np.load(file_path + '_adv.npy', allow_pickle=False)
        fpred = np.load(file_path + '_pred.npy', allow_pickle=False)
        fx = np.load(file_path + '_x.npy', allow_pickle=False)
        fy = np.load(file_path + '_y.npy', allow_pickle=False)

        self.assertTrue(np.allclose(fadv, adv))
        self.assertTrue(np.allclose(fx, x))
        self.assertTrue(np.array_equal(pred, fpred))
        self.assertTrue(np.array_equal(y, fy))


if __name__ == '__main__':
    unittest.main()
