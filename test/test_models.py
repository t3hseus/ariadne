import torch
import unittest

from unittest import TestCase
from ariadne.models import TrackNETv2


class TrackNETv2Test(TestCase):
    _batch_size = 64
    _input_features = 4
    _max_length = 6

    def _create_inputs(self):
        lengths = torch.randint(
            2, self._max_length, (self._batch_size,))
        lengths, _ = torch.sort(lengths, descending=True)
        tensors = [torch.randn((s, self._input_features)) for s in lengths]
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return padded, lengths

    def _do_forward(self):
        inputs, input_lengths = self._create_inputs()
        model = TrackNETv2(self._input_features)
        return model(inputs, input_lengths)

    def test_wrong_rnn_type(self):
        self.assertRaises(
            ValueError,
            lambda: TrackNETv2(rnn_type='l')
        )

    def test_output_shapes(self):
        xy, r1_r2 = self._do_forward()
        self.assertEqual(xy.shape, r1_r2.shape)
        self.assertEqual(list(xy.shape), [self._batch_size, 2])

    def test_predicted_radius_range(self):
        xy, r1_r2 = self._do_forward()
        self.assertGreater(torch.min(r1_r2).item(), 0)


if __name__ == '__main__':
    unittest.main()