import math
import torch 
import unittest
import numpy as np

from unittest import TestCase
from ariadne.tracknet_v2.metrics import (
    ellipse_area,
    point_in_ellipse,
    calc_metrics
)


class EllipseAreaTest(TestCase):
    def test_incorrect_shape(self):
        preds = torch.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, '4-dimensional'):
            ellipse_area(preds)

    def test_output(self):
        preds = torch.tensor([
            [0., 0., 1., 1.],
            [0., 0., 0.5, 0.5],
            [0., 0., 0., 100.]
        ])
        expected_output = torch.tensor([
            math.pi,
            0.25 * math.pi,
            0.
        ])

        for i in range(preds.size(0)):
            output = ellipse_area(preds[i:i+1])
            torch.testing.assert_allclose(output, expected_output[i])
        
        output = ellipse_area(preds)
        torch.testing.assert_allclose(output, expected_output.mean())


class PointInEllipseTest(TestCase):
    def test_incorrect_shape(self):
        target = torch.zeros((2, 2))
        preds = torch.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, 'Prediction must be 4-dimensional'):
            point_in_ellipse(preds, target)

        target = torch.zeros((2, 3))
        preds = torch.zeros((2, 4))
        with self.assertRaisesRegex(ValueError, 'Target must be 2-dimensional'):
            point_in_ellipse(preds, target)

        target = torch.zeros((2, 2))
        preds = torch.zeros((3, 4))
        with self.assertRaisesRegex(ValueError, 'Shape mismatch! Number of samples'):
            point_in_ellipse(preds, target)

    def test_output(self):
        preds = torch.tensor([
            [0., 0., 1., 1.],
            [0., -1., 1., 1.],
            [-1., 0., 1., 1.],
            [-0.5, -0.5, 1., 1.],
            [-1., -1., 1., 1.]
        ])
        target = torch.zeros((len(preds), 2))
        expected_output = [1, 1, 1, 1, 0]
        output = point_in_ellipse(preds, target)
        self.assertListEqual(output, expected_output)


class CalcMetricsTest(TestCase):
    def _create_model(self):
        """Dummy model. Takes last x, y 
        coords and predict ellipse with params
        (x, y, r1, r2) = (x+1, y+1, 1, 1)
        """
        def _model(inputs):
            # last x, y coords
            preds = inputs[:, -1, :2] + 1
            semiaxes = torch.ones((preds.size(0), 2))
            return torch.cat([preds, semiaxes], dim=1)

        def _model_eval():
            pass

        model = _model
        model.device = 'cpu'
        model.eval = _model_eval
        return model

    def test_incorrect_shape(self):
        model = self._create_model()
        inputs = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'Input array must be 3-dimensional'):
            calc_metrics(inputs, model)

        inputs = np.zeros((1, 2, 2))
        with self.assertRaisesRegex(ValueError, 'second dimension must be greater than 2'):
            calc_metrics(inputs, model)
        
        inputs = np.zeros((1, 5, 3))
        with self.assertRaisesRegex(ValueError, 'Invalid tracklen, must be >2'):
            calc_metrics(inputs, model, tracklen=2)

        with self.assertRaisesRegex(ValueError, 'Tracklen can`t be greater than number'):
            calc_metrics(inputs, model, tracklen=7)


    def test_output(self):

        def _assert_output(output, expected_output):
            self.assertAlmostEqual(output[0], expected_output[0])
            self.assertAlmostEqual(output[1], expected_output[1])

        model = self._create_model()
        inputs = np.tile(np.arange(5, dtype=np.float32), [3, 1]).T
        inputs = np.stack([inputs, inputs]) # batch of two equal samples
        expected_output = (1.0, 1.0)
        output = calc_metrics(inputs, model)
        _assert_output(output, expected_output)
        # miss last hit for the first track
        inputs[0, -1] += 3.0
        expected_output = (0.5, 0.9)
        output = calc_metrics(inputs, model)
        _assert_output(output, expected_output)


if __name__ == '__main__':
    unittest.main()


        