import logging
import os
import unittest

import numpy as np

from aad.datasets import (DATASET_LIST, DataContainer, get_sample_mean,
                          get_sample_std)
from aad.utils import get_data_path, master_seed, swap_image_channel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SEED = 4096


class TestDataContainer(unittest.TestCase):
    """Test DataContainer class"""

    @classmethod
    def init_datacontainer(cls, name):
        x = DATASET_LIST[name]
        path = get_data_path()
        dc = DataContainer(x, path)
        dc(shuffle=False, normalize=True, size_train=0.5)
        return dc

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
        self.assertTrue(os.path.exists(get_data_path()))

    def test_DataContainer_MNIST(self):
        dataname = 'MNIST'
        p = {'range': (0., 1.),
             'num_classes': 10,
             'type': 'image',
             'dim': (1, 28, 28),
             'mean': [0.13066046],
             'std': [0.30150425],
             'train_shape': (60000, 28, 28, 1),
             'test_shape': (10000, 28, 28, 1)}

        dc = self.init_datacontainer(dataname)

        # basic props
        r = dc.data_range
        self.assertEqual(r, p['range'])
        name = dc.name
        self.assertEqual(name, dataname)
        m = dc.num_classes
        self.assertEqual(m, p['num_classes'])
        dtype = dc.data_type
        self.assertEqual(dtype, p['type'])
        dim = dc.dim_data
        self.assertEqual(dim, p['dim'])
        mu = dc.train_mean
        np.testing.assert_array_almost_equal(mu, p['mean'])
        std = dc.train_std
        np.testing.assert_array_almost_equal(std, p['std'])

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_CIFAR10(self):
        dataname = 'CIFAR10'
        p = {
            'range': (0., 1.),
            'num_classes': 10,
            'type': 'image',
            'dim': (3, 32, 32),
            'mean': [0.49139947, 0.48215836, 0.44653094],
            'std': [0.20230092, 0.1994128, 0.20096162],
            'train_shape': (50000, 32, 32, 3),
            'test_shape': (10000, 32, 32, 3),
        }

        dc = self.init_datacontainer(dataname)

        # basic props
        mu = dc.train_mean
        np.testing.assert_array_almost_equal(mu, p['mean'])
        std = dc.train_std
        np.testing.assert_array_almost_equal(std, p['std'])

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_SVHN(self):
        dataname = 'SVHN'
        p = {
            'range': (0., 1.),
            'num_classes': 10,
            'type': 'image',
            'dim': (3, 32, 32),
            'mean': [0.43768215, 0.4437698, 0.47280422],
            'std': [0.12008651, 0.12313706, 0.10520393],
            'train_shape': (73257, 32, 32, 3),
            'test_shape': (26032, 32, 32, 3),
        }

        dc = self.init_datacontainer(dataname)

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = swap_image_channel(x_pt.cpu().detach().numpy())
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_BankNote(self):
        dataname = 'BankNote'
        p = {
            'range': [
                [-7.04209995, -13.7730999, -5.28609991, -8.54819965],
                [6.82480001, 12.95160007, 17.92740059, 2.44950008]],
            'num_classes': 2,
            'type': 'numeric',
            'dim': (4,),
            'mean': [0.67082083, 0.67537272, 0.26181248, 0.67141819],
            'std': [0.14560266, 0.19260927, 0.14039186, 0.19495234],
            'train_shape': (686, 4),
            'test_shape': (686, 4),
        }

        dc = self.init_datacontainer(dataname)

        # basic props
        r = np.array(dc.data_range)
        np.testing.assert_array_almost_equal(r, p['range'])
        name = dc.name
        self.assertEqual(name, dataname)
        m = dc.num_classes
        self.assertEqual(m, p['num_classes'])
        dtype = dc.data_type
        self.assertEqual(dtype, p['type'])
        dim = dc.dim_data
        self.assertEqual(dim, p['dim'])
        mu = dc.train_mean
        np.testing.assert_array_almost_equal(mu, p['mean'])
        std = dc.train_std
        np.testing.assert_array_almost_equal(std, p['std'])

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_BreastCancerWisconsin(self):
        dataname = 'BreastCancerWisconsin'
        p = {
            'dim': (30,),
            'train_shape': (284, 30),
            'test_shape': (285, 30),
        }

        dc = self.init_datacontainer(dataname)

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_HTRU2(self):
        dataname = 'HTRU2'
        p = {
            'dim': (8,),
            'train_shape': (8949, 8),
            'test_shape': (8949, 8),
        }

        dc = self.init_datacontainer(dataname)

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_Iris(self):
        dataname = 'Iris'
        p = {
            'dim': (4,),
            'train_shape': (75, 4),
            'test_shape': (75, 4),
        }

        dc = self.init_datacontainer(dataname)

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_DataContainer_WheatSeed(self):
        dataname = 'WheatSeed'
        p = {
            'dim': (7,),
            'train_shape': (105, 7),
            'test_shape': (105, 7),
        }

        dc = self.init_datacontainer(dataname)

        # train set
        train_np = dc.x_train
        self.assertEqual(train_np.shape, p['train_shape'])
        y_np = dc.y_train
        self.assertEqual(y_np.shape, (p['train_shape'][0],))
        train_loader = dc.get_dataloader(
            batch_size=8, is_train=True, shuffle=False)
        self.assertEqual(len(train_loader.dataset), p['train_shape'][0])
        x_pt, y_pt = next(iter(train_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = train_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

        # test set
        test_np = dc.x_test
        self.assertEqual(test_np.shape, p['test_shape'])
        y_np = dc.y_test
        self.assertEqual(y_np.shape, (p['test_shape'][0],))
        test_loader = dc.get_dataloader(
            batch_size=8, is_train=False, shuffle=False)
        self.assertEqual(len(test_loader.dataset), p['test_shape'][0])
        x_pt, y_pt = next(iter(test_loader))
        self.assertEqual(x_pt.size(), tuple([8]+list(p['dim'])))
        self.assertEqual(y_pt.size(), (8,))
        x1 = test_np[:8]
        x2 = x_pt.cpu().detach().numpy()
        np.testing.assert_equal(x1, x2)
        y1 = y_np[:8]
        y2 = y_pt[:8].cpu().detach().numpy()
        np.testing.assert_equal(y1, y2)

    def test_cross_validation(self):
        dataname = 'Iris'
        data_lookup = DATASET_LIST[dataname]
        path = get_data_path()
        dc = DataContainer(data_lookup, path)
        num_fold = 4
        dc(shuffle=False, normalize=True, cross_validation_fold=num_fold)
        part_size = 150 // num_fold

        x0, y0 = dc.get_one_fold_np(0)
        self.assertEqual(x0.shape, (part_size, 4))
        self.assertEqual(y0.shape, (part_size,))

        x3, y3 = dc.get_one_fold_np(3)
        self.assertEqual(x3.shape, (39, 4))
        self.assertEqual(y3.shape, (39,))

        self.assertTrue(not np.equal(x0[:10], x3[:10]).all())


if __name__ == '__main__':
    unittest.main()
