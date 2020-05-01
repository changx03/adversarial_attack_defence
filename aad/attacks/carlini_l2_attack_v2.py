"""
This module implements the Carlini and Wagner L2 attack.
"""
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets import GenericDataset
from ..utils import get_range, swap_image_channel
from .attack_container import AttackContainer

logger = logging.getLogger(__name__)


class CarliniL2V2Container(AttackContainer):
    """
    Class preforms Carlini and Wagner L2 attacks.
    """

    # np.inf does NOT play well with pytorch. 1e10 was used in carlini's implementation
    INF = 1e10

    def __init__(self,
                 model_container,
                 targeted=False,
                 learning_rate=1e-2,
                 binary_search_steps=9,
                 max_iter=1000,
                 confidence=0.0,
                 initial_const=1e-3,
                 c_range=(0, 1e10),
                 abort_early=True,
                 batch_size=32,
                 clip_values=None):
        """
        Create an instance of Carlini and Wagner L2-norm attack Container.

        targeted : bool
            Should we target one specific class? or just be wrong?
        learning_rate : float
            Larger values converge faster to less accurate results
        binary_search_steps : int
            Number of times to adjust the constant with binary search.
        max_iter : int
            Number of iterations to perform gradient descent
        confidence : float
            How strong the adversarial example should be. The parameter kappa in the paper
        initial_const : float
            The initial constant c_multiplier to pick as a first guess
        c_range : tuple, optional
            The lower and upper bounds for c_multiplier
        abort_early : bool
            If we stop improving, abort gradient descent early
        batch_size : int
            The size of a mini-batch.
        clip_values : tuple
            The clipping lower bound and upper bound for adversarial examples.
        """
        super(CarliniL2V2Container, self).__init__(model_container)

        dc = self.model_container.data_container
        data = dc.x_train
        dmax = np.max(data)
        dmin = np.min(data)
        if dmax > 1.0 or dmin < 0.0:
            logger.warning(
                'The data may not normalised. Consider using a normalised dataset.')

        if clip_values is None:
            clip_values = get_range(dc.x_train, dc.data_type == 'image')

        self._params = {
            'targeted': targeted,
            'learning_rate': learning_rate,
            'binary_search_steps': binary_search_steps,
            'max_iter': max_iter,
            'initial_const': initial_const,
            'c_range': c_range,
            'clip_values': clip_values,
            'abort_early': abort_early,
            'batch_size': batch_size,
        }

        self.confidence = confidence

    def generate(self, count=1000, use_testset=True, x=None, targets=None, **kwargs):
        """
        Generate adversarial examples.

        Parameters
        ----------
        count : int
            The number of adversarial examples will be generated from the test set. This parameter will not be used
            when 
        use_testset : bool
            Use test set to generate adversarial examples.
        x : numpy.ndarray, optional
            The data for generating adversarial examples. If this parameter is not null, `count` and `use_testset` will
            be ignored.
        targets : numpy.ndarray, optional
            The expected labels for targeted attack.

        Returns
        -------
        adv : numpy.ndarray
            The adversarial examples which have same shape as x.
        pred_adv :  : numpy.ndarray
            The predictions of adv. examples.
        x_clean : numpy.ndarray
            The clean inputs.
        pred_clean : numpy.ndarray
            The prediction of clean inputs.
        """
        assert use_testset or x is not None

        since = time.time()
        # parameters should able to set before training
        self.set_params(**kwargs)

        dc = self.model_container.data_container
        # handle the situation where testset has less samples than we want
        if use_testset and len(dc.x_test) < count:
            count = len(dc.x_test)

        if use_testset:
            x = np.copy(dc.x_test[:count])
            y = np.copy(dc.y_test[:count])
            acc = self.model_container.evaluate(x, y)
            logger.info('Accuracy on clean set: %f', acc)
        else:
            x = np.copy(x)

        # handle (h, w, c) to (c, h, w)
        data_type = self.model_container.data_container.data_type
        if data_type == 'image' and x.shape[1] not in (1, 3):
            xx = swap_image_channel(x)
        else:
            xx = x

        adv = self._generate(xx, targets)
        pred_adv, pred_clean = self.predict(adv, xx)

        # ensure the outputs and inputs have same shape
        if x.shape != adv.shape:
            adv = swap_image_channel(adv)
        time_elapsed = time.time() - since
        logger.info('Time to complete training %d adv. examples: %dm %.3fs',
                    count, int(time_elapsed // 60), time_elapsed % 60)
        return adv, pred_adv, x, pred_clean

    def _generate(self, inputs, targets=None):
        num_advs = len(inputs)
        num_classes = self.model_container.data_container.num_classes
        device = self.model_container.device
        c_range = self._params['c_range']
        initial_const = self._params['initial_const']
        batch_size = self._params['batch_size']
        lr = self._params['learning_rate']
        binary_search_steps = self._params['binary_search_steps']
        repeat = binary_search_steps >= 10
        max_iter = self._params['max_iter']
        abort_early = self._params['abort_early']

        # prepare data
        # Assume the predictions on clean inputs are correct.
        labels = self.model_container.predict(inputs)
        dataset = GenericDataset(inputs, labels)
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=0)

        full_input_np = np.zeros_like(inputs, dtype=np.float32)
        full_adv_np = np.zeros_like(inputs, dtype=np.float32)
        full_l2_np = 1e9 * np.ones(num_advs, dtype=np.float32)
        full_label_np = -np.ones(num_advs, dtype=np.int64)
        full_pred_np = -np.ones(num_advs, dtype=np.int64)

        count = 0  # only count the image can be classified correct
        self._log_time_start()
        for x, y in dataloader:
            since = time.time()

            x = x.to(device)
            y = y.to(device)
            batch_size = len(x)

            # c is the lagrange multiplier for optimization objective
            lower_bounds_np = np.ones(
                batch_size, dtype=np.float32) * c_range[0]
            c_np = np.ones(batch_size, dtype=np.float32) * initial_const
            upper_bounds_np = np.ones(
                batch_size, dtype=np.float32) * c_range[1]

            # overall results
            o_best_l2_np = np.ones(batch_size, dtype=np.float32) * self.INF
            o_best_pred_np = -np.ones(batch_size, dtype=np.int64)
            o_best_adv = torch.zeros_like(x)  # uses same device as x

            # we optimize over the tanh-space
            x_tanh = self._to_tanh(x, device)
            assert x_tanh.size() == x.size()

            # NOTE: testing untargeted attack here!
            targets = y
            # y in one-hot encoding
            targets_oh = self._onehot_encoding(targets)
            assert targets_oh.size() == (batch_size, num_classes)

            # the perturbation variable to optimize (In Carlini's code it's denoted as `modifier`)
            pert_tanh = torch.zeros_like(
                x, requires_grad=True)  # uses same device as x
            assert device == 'cpu' or pert_tanh.is_cuda

            # we retrain it for every batch
            optimizer = torch.optim.Adam([pert_tanh], lr=lr)

            for sstep in range(binary_search_steps):
                # at least try upper bound once
                if repeat and sstep == binary_search_steps - 1:
                    c_np = upper_bounds_np

                c = torch.from_numpy(c_np)
                c = c.to(device)

                best_l2_np = np.ones(batch_size, dtype=np.float32) * self.INF
                best_pred_np = -np.ones(batch_size, dtype=np.int64)

                # previous (summed) batch loss, to be used in early stopping policy
                prev_batch_loss = self.INF  # type: float

                # optimization step
                for ostep in range(max_iter):
                    loss, l2_norms, adv_outputs, advs = self._optimize(
                        optimizer, x_tanh, pert_tanh, targets_oh, c)

                    # check if we should abort search if we're getting nowhere
                    if abort_early and ostep % (max_iter//10) == 0:
                        loss = loss.cpu().detach().item()
                        assert type(prev_batch_loss) == type(loss)
                        if loss > prev_batch_loss * (1-1e-4):
                            break
                        prev_batch_loss = loss  # only check it 10 times

                    # update result
                    adv_outputs_np = adv_outputs.cpu().detach().numpy()
                    targets_np = targets.cpu().detach().numpy()

                    # compensate outputs with parameter confidence
                    adv_outputs_np = self._compensate_confidence(
                        adv_outputs_np, targets_np)
                    adv_predictions_np = np.argmax(adv_outputs_np, axis=1)

                    for i in range(batch_size):
                        i_l2 = l2_norms[i].item()
                        i_adv_pred = adv_predictions_np[i]
                        i_target = targets_np[i]
                        i_adv = advs[i]  # a tensor

                        if self._does_attack_success(i_adv_pred, i_target):
                            if i_l2 < best_l2_np[i]:
                                best_l2_np[i] = i_l2
                                best_pred_np[i] = i_adv_pred

                            if i_l2 < o_best_l2_np[i]:
                                o_best_l2_np[i] = i_l2
                                o_best_pred_np[i] = i_adv_pred
                                o_best_adv[i] = i_adv

                # binary search for c
                for i in range(batch_size):
                    i_target = targets_np[i]
                    assert best_pred_np[i] == - \
                        1 or self._does_attack_success(
                            best_pred_np[i], i_target)
                    assert o_best_pred_np[i] == - \
                        1 or self._does_attack_success(
                            o_best_pred_np[i], i_target)

                    if best_pred_np[i] != -1:  # successful, try lower `c` value
                        # update upper bound, and divide c by 2
                        upper_bounds_np[i] = min(upper_bounds_np[i], c_np[i])
                        # 1e9 was used in carlini's implementation
                        if upper_bounds_np[i] < c_range[1] * 0.1:
                            c_np[i] = (lower_bounds_np[i] +
                                       upper_bounds_np[i]) / 2.

                    else:  # failure, try larger `c` value
                        # either multiply by 10 if no solution found yet
                        # or do binary search with the known upper bound
                        lower_bounds_np[i] = max(lower_bounds_np[i], c_np[i])

                        # 1e9 was used in carlini's implementation
                        if upper_bounds_np[i] < c_range[1] * 0.1:
                            c_np[i] = (lower_bounds_np[i] +
                                       upper_bounds_np[i]) / 2.
                        else:
                            c_np[i] *= 10

            # save results
            full_l2_np[count: count+batch_size] = o_best_l2_np
            full_label_np[count: count+batch_size] = y.cpu().detach().numpy()
            full_pred_np[count: count+batch_size] = o_best_pred_np
            full_input_np[count: count+batch_size] = x.cpu().detach().numpy()
            full_adv_np[count: count +
                        batch_size] = o_best_adv.cpu().detach().numpy()

            # display logs
            time_elapsed = time.time() - since
            mean_l2 = np.mean(o_best_l2_np)
            debug_str = '[{:4d}/{:4d}] - {:2.0f}m {:2.1f}s - L2 mean: {:.4f}'.format(
                count,
                num_advs,
                time_elapsed // 60,
                time_elapsed % 60,
                mean_l2)
            logger.debug(debug_str)

            count += batch_size

        self._log_time_end('Carlini & Wagner Attack L2')
        return full_adv_np

    @staticmethod
    def _arctanh(x, epsilon=1e-6):
        assert isinstance(x, torch.Tensor)

        # to enhance numeric stability. avoiding divide by zero
        x = x * (1-epsilon)
        return 0.5 * torch.log((1.+x) / (1.-x))

    def _to_tanh(self, x, device=None):
        assert isinstance(x, torch.Tensor)
        bound = self._params['clip_values']
        dmin = torch.tensor(bound[0], dtype=torch.float32)
        dmax = torch.tensor(bound[1], dtype=torch.float32)
        if device is not None:
            dmin = dmin.to(device)
            dmax = dmax.to(device)
        box_mul = (dmax - dmin) * .5
        box_plus = (dmax + dmin) * .5
        return self._arctanh((x - box_plus) / box_mul)

    def _from_tanh(self, w, device=None):
        assert isinstance(w, torch.Tensor)
        bound = self._params['clip_values']
        dmin = torch.tensor(bound[0], dtype=torch.float32)
        dmax = torch.tensor(bound[1], dtype=torch.float32)
        if device is not None:
            dmin = dmin.to(device)
            dmax = dmax.to(device)
        box_mul = (dmax - dmin) * .5
        box_plus = (dmax + dmin) * .5
        return torch.tanh(w)*box_mul + box_plus

    def _onehot_encoding(self, labels):
        num_classes = self.model_container.data_container.num_classes
        device = self.model_container.device

        assert isinstance(labels, torch.Tensor), type(labels)
        assert labels.max().item(
        ) < num_classes, f'{labels.max()} > {num_classes}'

        labels_t = labels.unsqueeze(1)
        y_onehot = torch.zeros(len(labels), num_classes, dtype=torch.int8)
        y_onehot = y_onehot.to(device)
        return y_onehot.scatter_(1, labels_t, 1)

    @staticmethod
    def _get_l2_norm(a, b, dim=1):
        assert isinstance(a, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        assert a.size() == b.size()

        return torch.norm(
            a.view(a.size(0), -1) - b.view(b.size(0), -1),
            dim=dim)

    def _optimize(self, optimizer, inputs_tanh, pert_tanh, targets_oh, const):
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(inputs_tanh, torch.Tensor)
        assert isinstance(pert_tanh, torch.Tensor)
        assert isinstance(targets_oh, torch.Tensor)
        assert isinstance(const, torch.Tensor)

        batch_size = inputs_tanh.size(0)
        is_targeted = self._params['targeted']
        confidence = self.confidence
        device = self.model_container.device

        assert const.size() == (batch_size,)

        optimizer.zero_grad()
        # the adversarial examples in image space
        advs = self._from_tanh(inputs_tanh + pert_tanh, device)
        # the clean images converted back from tanh space
        inputs = self._from_tanh(inputs_tanh, device)

        # The softmax is stripped out from this model.
        model = self.model_container.model
        adv_outputs = model(advs)
        if torch.equal(
                torch.round(adv_outputs.sum(1)),
                torch.ones(len(adv_outputs)).to(device)):
            raise ValueError(
                'The score from the model should NOT be probability!')

        l2_norms = self._get_l2_norm(advs, inputs)
        assert l2_norms.size() == (batch_size,)

        target_outputs = torch.sum(targets_oh * adv_outputs, 1)
        other_outputs = torch.max(
            (1.0-targets_oh)*adv_outputs - targets_oh*self.INF, 1)[0]

        if is_targeted:
            f_loss = torch.clamp(
                other_outputs - target_outputs + confidence, min=0.)
        else:
            f_loss = torch.clamp(
                target_outputs - other_outputs + confidence, min=0.)
        assert f_loss.size() == (batch_size,)

        loss = torch.sum(l2_norms + const * f_loss)
        loss.backward()
        optimizer.step()

        return loss, l2_norms, adv_outputs, advs

    def _compensate_confidence(self, outputs, targets):
        assert type(outputs) == np.ndarray
        assert type(targets) == np.ndarray

        is_targeted = self._params['targeted']
        confidence = self.confidence

        outputs_comp = np.copy(outputs)
        indices = np.arange(targets.shape[0])
        if is_targeted:
            outputs_comp[indices, targets] -= confidence
        else:
            outputs_comp[indices, targets] += confidence

        return outputs_comp

    def _does_attack_success(self, pred, label):
        is_targeted = self._params['targeted']
        if is_targeted:
            return int(pred) == int(label)  # match the target label
        else:
            return int(pred) != int(label)  # anyting other than the true label
