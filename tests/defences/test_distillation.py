import logging
import os
import unittest

import numpy as np
import torch

from aad.defences import DistillationContainer
from aad.utils import master_seed

logger = logging.getLogger(__name__)

SEED = 4096
BATCH_SIZE = 128
NAME = 'MNIST'
MAX_EPOCHS = 5


class TestDistillation(unittest.TestCase):
    """Testing Distillation as Defence."""
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        master_seed(SEED)

    def test_loss_func(self):
        x = torch.randn((3, 2))
        target = torch.softmax(x, dim=1)
        loss = DistillationContainer.smooth_nlloss(x, target)
        self.assertAlmostEqual(loss.item(), 0.5781, places=4)


if __name__ == '__main__':
    unittest.main()
