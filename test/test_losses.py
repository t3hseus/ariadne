import torch
import unittest

from unittest import TestCase
from ariadne.tracknet_v2.loss import (
    PointInEllipseLoss,
    EllipseSquareLoss,
    TrackNetLoss
)


class EllipseSquareLossTest(TestCase):
    _loss_func = EllipseSquareLoss()

    def test_output(self):
        preds = torch.zeros((5, 1, 4)).float()
        preds[:, 0, 2] = torch.tensor([0, 0, 1, 0.9, 2.5])
        preds[:, 0, 3] = torch.tensor([0, 1, 1, 0.3, 2])
        expected_output = torch.tensor([0., 0., 1., 0.27, 5.]).unsqueeze(1)

        for i in range(preds.size(0)):
            output = self._loss_func(preds[i:i+1])
            torch.testing.assert_allclose(output, expected_output[i])

        output = self._loss_func(preds)
        torch.testing.assert_allclose(output, expected_output)

    def test_incorrect_shape(self):
        preds = torch.zeros((5, 6, 3))
        with self.assertRaisesRegex(ValueError, 'Prediction must be 4-dimensional'):
            self._loss_func(preds)


class PointInEllipseLossTest(TestCase):
    _loss_func = PointInEllipseLoss()

    def test_incorrect_shape(self):
        timesteps = 6
        target = torch.zeros((2, timesteps, 2))
        preds = torch.zeros((3, timesteps, 4))
        with self.assertRaisesRegex(ValueError, 'Shape mismatch! Number of samples'):
            self._loss_func(preds, target)

        target = torch.zeros((2, timesteps, 2))
        preds = torch.zeros((2, timesteps, 3))
        with self.assertRaisesRegex(ValueError, 'Prediction must be 4-dimensional'):
            self._loss_func(preds, target)

        target = torch.zeros((2, timesteps, 3))
        preds = torch.zeros((2, timesteps, 4))
        with self.assertRaisesRegex(ValueError, 'Target must be 2-dimensional'):
            self._loss_func(preds, target)

    def test_output(self):
        timesteps = 7
        preds = torch.empty(2, timesteps, 4)
        preds[:, 0] = torch.tensor([0., 0., 1., 1.])
        preds[:, 1] = torch.tensor([0.5, 0., 1., 1.])
        preds[:, 2] = torch.tensor([0., 0.5, 1., 1.])
        preds[:, 3] = torch.tensor([1., 0., 1., 1.])
        preds[:, 4] = torch.tensor([0., 1., 1., 1.])
        preds[:, 5] = torch.tensor([1., 1., 1., 1.])
        preds[:, 6] = torch.tensor([2., 2., 1., 1.])
        target = torch.zeros((len(preds), timesteps, 2)).float()
        expected_output = torch.tensor([
            [0., 0.5, 0.5, 1., 1., 2.**0.5, 8 ** 0.5],
            [0., 0.5, 0.5, 1., 1., 2.**0.5, 8 ** 0.5]
        ])

        for i in range(preds.size(0)):
            output = self._loss_func(preds[:, i:i+1], target[:, i:i+1])
            torch.testing.assert_allclose(output, expected_output[:, i:i+1])

        output = self._loss_func(preds, target)
        torch.testing.assert_allclose(output, expected_output)


class TrackNetLossTest(TestCase):
    def test_invalid_alpha(self):
        with self.assertRaisesRegex(ValueError, 'Weighting factor alpha must be'):
            loss = TrackNetLoss(alpha=-1)
        with self.assertRaisesRegex(ValueError, 'Weighting factor alpha must be'):
            loss = TrackNetLoss(alpha=1.01)

    def test_output(self):
        loss_func = TrackNetLoss(alpha=0.5)
        n_examples, timesteps = 3, 5
        preds = torch.empty(n_examples, timesteps, 4)
        preds[0] = torch.tensor([0., 0., 1., 1.])
        preds[1] = torch.tensor([0., 0.5, 1., 1.])
        preds[2] = torch.tensor([2., 2., 1., 1.])
        target = torch.zeros((len(preds), timesteps, 2)).float()

        expected_output = torch.tensor([
            0.5,
            0.75,
            (8 ** 0.5) * 0.5 + 0.5
        ])
        expected_output = torch.repeat_interleave(expected_output, timesteps).view(n_examples, timesteps)

        for i in range(preds.size(0)):
            output = loss_func(preds[i:i+1], target[i:i+1])
            torch.testing.assert_allclose(output, expected_output[i].mean())

        output = loss_func(preds, target)
        torch.testing.assert_allclose(output, expected_output.mean())

        # test with mask
        mask = torch.ones(n_examples, timesteps).bool()
        # mask first two and last timestep
        mask[:, :1] = False
        mask[:, -1] = False
        # mean of (0.5, 0.75 and ((8 ** 0.5) * 0.5 + 0.5))
        expected_result = expected_output[:, 0].mean()
        # modify three timesteps in every sample in the batch
        # to be like the following
        # [
        #     ... first timestep ...
        #     [0., 0., 1., 1.]
        #     [0., 0.5, 1., 1.]
        #     [2., 2., 1., 1.]
        #     ... last timestep ...
        # ]
        preds[:, 1:-1] = preds[:, 0, :]
        # calculate output
        output = loss_func(preds, target, mask)
        torch.testing.assert_allclose(output, expected_result)


if __name__ == '__main__':
    unittest.main()
