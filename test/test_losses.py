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
        preds = torch.zeros((5, 4)).float()
        preds[:, 2] = torch.tensor([0, 0, 1, 0.9, 2.5])
        preds[:, 3] = torch.tensor([0, 1, 1, 0.3, 2])
        expected_output = torch.tensor([0., 0., 1., 0.27, 5.])

        for i in range(preds.size(0)):
            output = self._loss_func(preds[i:i+1])
            torch.testing.assert_allclose(output, expected_output[i])
        
        output = self._loss_func(preds)
        torch.testing.assert_allclose(output, expected_output.mean())

    def test_incorrect_shape(self):
        preds = torch.zeros((5, 3))
        with self.assertRaisesRegex(ValueError, 'Prediction must be 4-dimensional'):
            self._loss_func(preds)


class PointInEllipseLossTest(TestCase):
    _loss_func = PointInEllipseLoss()

    def test_incorrect_shape(self):
        target = torch.zeros((2, 2))
        preds = torch.zeros((3, 4))
        with self.assertRaisesRegex(ValueError, 'Shape mismatch! Number of samples'):
            self._loss_func(preds, target)

        target = torch.zeros((2, 2))
        preds = torch.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, 'Prediction must be 4-dimensional'):
            self._loss_func(preds, target)

        target = torch.zeros((2, 3))
        preds = torch.zeros((2, 4))
        with self.assertRaisesRegex(ValueError, 'Target must be 2-dimensional'):
            self._loss_func(preds, target)

    def test_output(self):
        preds = torch.tensor([
            [0., 0., 1., 1.],
            [0.5, 0., 1., 1.],
            [0., 0.5, 1., 1.],
            [1., 0., 1., 1.],
            [0., 1., 1., 1.],
            [1., 1., 1., 1.],
            [2., 2., 1., 1.]
        ])
        target = torch.zeros((len(preds), 2)).float()
        expected_output = torch.tensor([0., 0.5, 0.5, 1., 1., 2.**0.5, 8 ** 0.5])

        for i in range(preds.size(0)):
            output = self._loss_func(preds[i:i+1], target[i:i+1])
            torch.testing.assert_allclose(output, expected_output[i])
        
        output = self._loss_func(preds, target)
        torch.testing.assert_allclose(output, expected_output.mean())


class TrackNetLossTest(TestCase):
    def test_invalid_alpha(self):
        with self.assertRaisesRegex(ValueError, 'Weighting factor alpha must be'):
            loss = TrackNetLoss(alpha=-1)
        with self.assertRaisesRegex(ValueError, 'Weighting factor alpha must be'):
            loss = TrackNetLoss(alpha=1.01)

    def test_output(self):
        loss_func = TrackNetLoss(alpha=0.5)
        preds = preds = torch.tensor([
            [0., 0., 1., 1.],
            [0., 0.5, 1., 1.],
            [2., 2., 1., 1.]
        ])
        target = torch.zeros((len(preds), 2)).float()
        expected_output = torch.tensor([
            0.5,
            0.75,
            (8 ** 0.5) * 0.5 + 0.5
        ])
        
        for i in range(preds.size(0)):
            output = loss_func(preds[i:i+1], target[i:i+1])
            torch.testing.assert_allclose(output, expected_output[i])
        
        output = loss_func(preds, target)
        torch.testing.assert_allclose(output, expected_output.mean())    


if __name__ == '__main__':
    unittest.main()
