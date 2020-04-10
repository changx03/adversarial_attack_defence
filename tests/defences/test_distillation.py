import logging
import os
import unittest

import numpy as np
import torch

from aad.defences import DistillationContainer
from aad.utils import master_seed, get_data_path, get_pt_model_filename
from aad.basemodels import ModelContainerPT, get_model
from aad.datasets import DATASET_LIST, DataContainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096
BATCH_SIZE = 128
NAME = 'MNIST'
MAX_EPOCHS = 30
TEMPERATURE = 2.0
MODEL_FILE = os.path.join('save', 'MnistCnnV2_MNIST_e50.pt')
ADV_FILE = os.path.join('save', 'MnistCnnV2_MNIST_FGSM_adv.npy')


class TestDistillation(unittest.TestCase):
    """Testing Distillation as Defence."""
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        Model = get_model('MnistCnnV2')
        model = Model()
        logger.info('Starting %s data container...', NAME)
        dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        dc()
        mc = ModelContainerPT(model, dc)
        mc.load(MODEL_FILE)
        accuracy = mc.evaluate(dc.data_test_np, dc.label_test_np)
        logger.info('Accuracy on test set: %f', accuracy)

        cls.distillation = DistillationContainer(
            mc, Model(), temperature=TEMPERATURE, pretrained=False)

        filename = get_pt_model_filename(
            model.__class__.__name__,
            NAME,
            str(MAX_EPOCHS) + 't' + str(int(TEMPERATURE*10)))
        filename = os.path.join('test', 'distill_' + filename)
        file_path = os.path.join('save', filename)
        if not os.path.exists(file_path):
            # Expected initial loss = -log(1/num_classes) = 2.3025850929940455'
            cls.distillation.fit(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
            cls.distillation.save(filename, overwrite=True)
        else:
            cls.distillation.load(file_path)

        smooth_mc = cls.distillation.get_def_model_container()
        accuracy = smooth_mc.evaluate(dc.data_test_np, dc.label_test_np)
        logger.info('Accuracy on test set: %f', accuracy)

    def setUp(self):
        master_seed(SEED)

    def test_loss_func(self):
        x = torch.randn((3, 2))
        target = torch.softmax(x, dim=1)
        loss = DistillationContainer.smooth_nlloss(x, target)
        self.assertAlmostEqual(loss.item(), 0.5781, places=4)

    def test_trainset(self):
        mc = self.distillation.get_def_model_container()
        x_train = mc.data_container.data_train_np
        score = mc.get_score(x_train)
        # prob = torch.softmax(torch.from_numpy(score) / TEMPERATURE, dim=1)
        prob = torch.softmax(torch.from_numpy(score), dim=1)
        softlabel_train = torch.from_numpy(mc.data_container.label_train_np)
        l2 = torch.mean((softlabel_train - prob).norm(dim=1))
        self.assertLessEqual(l2.item(), 0.0274)

    def test_detect(self):
        postfix = ['adv', 'pred', 'x', 'y']
        data_files = [ADV_FILE.replace('_adv', '_' + s) for s in postfix]

        adv = np.load(data_files[0], allow_pickle=False)
        pred = np.load(data_files[1], allow_pickle=False)
        x = np.load(data_files[2], allow_pickle=False)
        y = np.load(data_files[3], allow_pickle=False)

        blocked_indices = self.distillation.detect(adv, pred)
        num_blocked = len(blocked_indices)
        self.assertGreaterEqual(num_blocked, 50)
        logger.info('blocked adversarial: %d', num_blocked)

        blocked_indices = self.distillation.detect(x, y)
        num_blocked = len(blocked_indices)
        self.assertLessEqual(num_blocked, 5)
        logger.info('blocked clean: %d', num_blocked)


if __name__ == '__main__':
    unittest.main()
