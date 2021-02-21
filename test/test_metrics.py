import gin
import math
import torch
import unittest
import numpy as np

from unittest import TestCase
import pytorch_lightning as pl
from ariadne.tracknet_v2.metrics import (
    ellipse_area,
    point_in_ellipse,
    calc_metrics
)
from ariadne.metrics import (
    accuracy,
    f1_score,
    recall,
    precision
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


class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        gin.clear_config()

    @staticmethod
    def etalon_recall(preds, target, threshold=0.5):
        preds = TestClassificationMetrics._format_preds(preds, target, threshold)
        result = preds[target == 1].float().mean()
        if torch.isnan(result):
            return torch.tensor(0.).float()
        return result

    @staticmethod
    def _format_preds(preds, target, threshold):
        if len(preds.shape) > len(target.shape):
            # softmax output
            preds = torch.argmax(preds, dim=1).bool()
        else:
            preds = preds > threshold
        return preds

    @staticmethod
    def etalon_precision(preds, target, threshold=0.5):
        preds = TestClassificationMetrics._format_preds(preds, target, threshold)
        result = target[preds].float().mean()
        if torch.isnan(result):
            return torch.tensor(1.).float()
        return result

    @staticmethod
    def etalon_accuracy(preds, target, threshold=0.5):
        preds = TestClassificationMetrics._format_preds(preds, target, threshold)
        return (preds == target).float().mean()

    @staticmethod
    def etalon_f1(preds, target, threshold=0.5):
        recall = TestClassificationMetrics.etalon_recall(preds, target, threshold)
        precision = TestClassificationMetrics.etalon_precision(preds, target, threshold)
        return 2 * (precision * recall) / (precision + recall)

    def _test_serial(self, target, test_inputs, metric_func):
        if metric_func is recall:
            etalon_func = self.etalon_recall
        elif metric_func is precision:
            etalon_func = self.etalon_precision
        elif metric_func is accuracy:
            etalon_func = self.etalon_accuracy
        elif metric_func is f1_score:
            etalon_func = self.etalon_f1
        elif isinstance(metric_func, pl.metrics.Recall):
            etalon_func = self.etalon_recall

        for preds, result in test_inputs:
            with self.subTest(preds=preds, target=target, result=result):
                ariadne_result = metric_func(preds, target)
                etalon_result = etalon_func(preds, target)
                self.assertEqual(ariadne_result, etalon_result)
                self.assertEqual(ariadne_result, result)

    def test_recall(self):
        # 1D case
        target = torch.tensor([1, 1, 1, 1]).int()
        test_inputs = [
            (torch.tensor([0., 0., 0., 0.]), 0.),
            (torch.tensor([0.6, 0.7, 0.99, 0.7]), 1.),
            (torch.tensor([0., 0.6, 0.6, 0.]), 0.5),
            (torch.tensor([0.2, 0.8, 0.45, 0.49]), 0.25)
        ]
        self._test_serial(target, test_inputs, recall)
        # 2D case
        target = target.view(2, 2)
        test_inputs = [(preds.view(2, 2), result) for preds, result in test_inputs]
        self._test_serial(target, test_inputs, recall)
        # softmax case
        test_inputs = [
            (torch.tensor([
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.8, 0.2], [0.9, 0.1]]
            ]), 0.),
            (torch.tensor([
                [[0.2, 0.8], [0.3, 0.7]],
                [[0.2, 0.8], [0.1, 0.9]]
            ]), 1.),
            (torch.tensor([
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.2, 0.8], [0.6, 0.4]]
            ]), 0.5),
            (torch.tensor([
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4]]
            ]), 0.25),
        ]
        # (N, F, C) -> (N, C, F) classes are second
        test_inputs = [(preds.transpose(2, 1), result) for preds, result in test_inputs]
        gin.bind_parameter('ariadne.metrics.recall.is_softmax', True)
        self._test_serial(target, test_inputs, recall)

    def test_class_recall(self):
        # 1D case
        target = torch.tensor([1, 1, 1, 1]).int()
        test_inputs = [
            (torch.tensor([0., 0., 0., 0.]), 0.),
            (torch.tensor([0.6, 0.7, 0.99, 0.7]), 1.),
            (torch.tensor([0., 0.6, 0.6, 0.]), 0.5),
            (torch.tensor([0.2, 0.8, 0.45, 0.49]), 0.25)
        ]
        recall_metric = pl.metrics.Recall(is_multiclass=False)
        self._test_serial(target, test_inputs, recall_metric)
        # 2D case
        target = target.view(2, 2)
        test_inputs = [(preds.view(2, 2), result) for preds, result in test_inputs]
        self._test_serial(target, test_inputs, recall_metric)
        # softmax case
        test_inputs = [
            (torch.tensor([
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.8, 0.2], [0.9, 0.1]]
            ]), 0.),
            (torch.tensor([
                [[0.2, 0.8], [0.3, 0.7]],
                [[0.2, 0.8], [0.1, 0.9]]
            ]), 1.),
            (torch.tensor([
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.2, 0.8], [0.6, 0.4]]
            ]), 0.5),
            (torch.tensor([
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4]]
            ]), 0.25),
        ]
        # (N, F, C) -> (N, C, F) classes are second
        test_inputs = [(preds.transpose(2, 1), result) for preds, result in test_inputs]
        recall_metric = pl.metrics.Recall(is_multiclass=True, mdmc_average='global')
        gin.bind_parameter('ariadne.metrics.recall.is_softmax', True)
        self._test_serial(target, test_inputs, recall_metric)

    def test_precision(self):
        # 1D case
        target = torch.tensor([1, 0, 0, 0]).int()
        test_inputs = [
            (torch.tensor([0.2, 0.9, 0.4, 0.9]), 0.),
            (torch.tensor([0.9, 0.0, 0.3, 0.4]), 1.),
            (torch.tensor([0.9, 0.9, 0.2, 0.4]), 0.5),
            (torch.tensor([0.9, 0.8, 0.7, 0.6]), 0.25),
        ]
        self._test_serial(target, test_inputs, precision)
        # 2D case
        target = target.view(2, 2)
        test_inputs = [(preds.view(2, 2), result) for preds, result in test_inputs]
        self._test_serial(target, test_inputs, precision)
        # softmax case
        test_inputs = [
            (torch.tensor([
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.8, 0.2], [0.1, 0.9]]
            ]), 0.),
            (torch.tensor([
                [[0.2, 0.8], [0.7, 0.3]],
                [[0.8, 0.2], [0.9, 0.1]]
            ]), 1.),
            (torch.tensor([
                [[0.1, 0.9], [0.1, 0.9]],
                [[0.8, 0.2], [0.6, 0.4]]
            ]), 0.5),
            (torch.tensor([
                [[0.1, 0.9], [0.2, 0.8]],
                [[0.3, 0.7], [0.4, 0.6]]
            ]), 0.25),
        ]
        # (N, F, C) -> (N, C, F) classes are second
        test_inputs = [(preds.transpose(2, 1), result) for preds, result in test_inputs]
        gin.bind_parameter('ariadne.metrics.precision.is_softmax', True)
        self._test_serial(target, test_inputs, precision)

    def test_accuracy(self):
        # 1D case
        target = torch.tensor([1, 0, 0, 1]).int()
        test_inputs = [
            (torch.tensor([0.2, 0.9, 0.8, 0.1]), 0.),
            (torch.tensor([0.9, 0.0, 0.3, 0.55]), 1.),
            (torch.tensor([0.9, 0.2, 0.9, 0.4]), 0.5),
            (torch.tensor([0.2, 0.2, 0.7, 0.49]), 0.25),
        ]
        self._test_serial(target, test_inputs, accuracy)
        # 2D case
        target = target.view(2, 2)
        test_inputs = [(preds.view(2, 2), result) for preds, result in test_inputs]
        self._test_serial(target, test_inputs, accuracy)
        # softmax case
        test_inputs = [
            (torch.tensor([
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.2, 0.8], [0.9, 0.1]]
            ]), 0.),
            (torch.tensor([
                [[0.2, 0.8], [0.7, 0.3]],
                [[0.8, 0.2], [0.1, 0.9]]
            ]), 1.),
            (torch.tensor([
                [[0.1, 0.9], [0.9, 0.1]],
                [[0.2, 0.8], [0.6, 0.4]]
            ]), 0.5),
            (torch.tensor([
                [[0.9, 0.1], [0.8, 0.2]],
                [[0.3, 0.7], [0.6, 0.4]]
            ]), 0.25),
        ]
        # (N, F, C) -> (N, C, F) classes are second
        test_inputs = [(preds.transpose(2, 1), result) for preds, result in test_inputs]
        gin.bind_parameter('ariadne.metrics.accuracy.is_softmax', True)
        self._test_serial(target, test_inputs, accuracy)

    def test_f1(self):
        # 1D case
        # 1D case
        target = torch.tensor([1, 1, 1, 1]).int()
        # F1 = 2 * (precision * recall) / (precision + recall)
        test_inputs = [
            (torch.tensor([0., 0., 0., 0.]), 0.), # recall=0, precision=1, f1=0
            (torch.tensor([0.6, 0.7, 0.99, 0.7]), 1.), # recall=1, precision=1, f1=1
            (torch.tensor([0., 0.6, 0.6, 0.]), 0.66666666), # recall=0.5, precision=1, f1=0.666
            (torch.tensor([0.2, 0.8, 0.45, 0.49]), 0.4) # recall=0.25, precision=1, f1=0.4
        ]
        self._test_serial(target, test_inputs, f1_score)
        # 2D case
        target = target.view(2, 2)
        test_inputs = [(preds.view(2, 2), result) for preds, result in test_inputs]
        self._test_serial(target, test_inputs, f1_score)
        # softmax case
        test_inputs = [
            (torch.tensor([
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.8, 0.2], [0.9, 0.1]]
            ]), 0.),
            (torch.tensor([
                [[0.2, 0.8], [0.3, 0.7]],
                [[0.2, 0.8], [0.1, 0.9]]
            ]), 1.),
            (torch.tensor([
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.2, 0.8], [0.6, 0.4]]
            ]), 0.66666666),
            (torch.tensor([
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4]]
            ]), 0.4),
        ]
        # (N, F, C) -> (N, C, F) classes are second
        test_inputs = [(preds.transpose(2, 1), result) for preds, result in test_inputs]
        gin.bind_parameter('ariadne.metrics.f1_score.is_softmax', True)
        self._test_serial(target, test_inputs, f1_score)


if __name__ == '__main__':
    unittest.main()
