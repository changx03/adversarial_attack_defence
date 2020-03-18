import logging
import unittest

from aad.basemodels import IrisNN, TorchModelContainer
from aad.datasets import DATASET_LIST, DataContainer
from aad.defences import ApplicabilityDomainContainer
from aad.utils import get_data_path, master_seed

logger = logging.getLogger(__name__)
SEED = 4096  # 2**12 = 4096
BATCH_SIZE = 128


class TestApplicabilityDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(SEED)

        NAME = 'Iris'
        logger.info(f'Starting {NAME} data container...')
        cls.dc = DataContainer(DATASET_LIST[NAME], get_data_path())
        cls.dc(shuffle=True)

        model = IrisNN()
        model_name = model.__class__.__name__
        logger.info(f'Using model: {model_name}')
        cls.mc = TorchModelContainer(model, cls.dc)
        cls.mc.fit(epochs=100, batch_size=BATCH_SIZE)

        cls.ad = ApplicabilityDomainContainer(
            cls.mc, k1=3, k2=6, confidence=1.0)

    def setUp(self):
        master_seed(SEED)

    def test_fit(self):
        result = self.ad.fit()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
