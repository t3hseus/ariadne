import unittest
import numpy as np
from . import original_model as orig
from .model import *



def _share_tnet_weights(tnet, tnet_orig):
    # conv
    tnet.features.conv1[0].load_state_dict(tnet_orig.conv1.state_dict())
    tnet.features.conv2[0].load_state_dict(tnet_orig.conv2.state_dict())
    tnet.features.conv3[0].load_state_dict(tnet_orig.conv3.state_dict())
    # conv batch norms
    tnet.features.conv1[1].load_state_dict(tnet_orig.bn1.state_dict())
    tnet.features.conv2[1].load_state_dict(tnet_orig.bn2.state_dict())
    tnet.features.conv3[1].load_state_dict(tnet_orig.bn3.state_dict())
    # linear layers
    tnet.features.fc1[0].load_state_dict(tnet_orig.fc1.state_dict())
    tnet.features.fc2[0].load_state_dict(tnet_orig.fc2.state_dict())
    # fc batchnorms
    tnet.features.fc1[1].load_state_dict(tnet_orig.bn4.state_dict())
    tnet.features.fc2[1].load_state_dict(tnet_orig.bn5.state_dict())
    # last layer
    tnet.transform.load_state_dict(tnet_orig.fc3.state_dict())


class TestTnet(unittest.TestCase):
    def test_output(self):
        tnet_orig = orig.Tnet(k=3)
        tnet = TNet(
            k=3,
            conv_layers=(64, 128, 1024),
            linear_layers=(512, 256),
            pooling='max'
        )

        # share parameters
        _share_tnet_weights(tnet, tnet_orig)

        t = torch.rand(32, 3, 500)
        with torch.no_grad():
            tnet_orig_out = tnet_orig(t)
            tnet_out = tnet(t)

        np.testing.assert_almost_equal(
            tnet_orig_out.numpy(),
            tnet_out.numpy()
        )


class TestTransform(unittest.TestCase):
    def test_output(self):
        transform_orig = orig.Transform()
        transform = Transform(
            input_transform=TNet(
                k=3,
                conv_layers=(64, 128, 1024),
                linear_layers=(512, 256)
            ),
            feature_transform=TNet(
                k=64,
                conv_layers=(64, 128, 1024),
                linear_layers=(512, 256)
            ),
            top_layers=(128, 1024)
        )
        # share weights between input transforms
        _share_tnet_weights(
            transform.input_transform,
            transform_orig.input_transform
        )
        # share weights between feature transforms
        _share_tnet_weights(
            transform.feature_transform,
            transform_orig.feature_transform
        )
        # share the remaining weights
        transform.input2features[0].load_state_dict(transform_orig.conv1.state_dict())
        transform.input2features[1].load_state_dict(transform_orig.bn1.state_dict())
        transform.top_layers.conv1[0].load_state_dict(transform_orig.conv2.state_dict())
        transform.top_layers.conv1[1].load_state_dict(transform_orig.bn2.state_dict())
        transform.top_layers.conv2[0].load_state_dict(transform_orig.conv3.state_dict())
        transform.top_layers.conv2[1].load_state_dict(transform_orig.bn3.state_dict())

        # test outputs
        t = torch.rand(32, 3, 500)

        with torch.no_grad():
            orig_out, orig_arr3x3, orig_arr64x64 = transform_orig(t)
            out, arr3x3, arr64x64 = transform(t)

        np.testing.assert_almost_equal(
            orig_out.numpy(),
            # remove local features from the output
            out[:, -orig_out.size(1):, 0].numpy()
        )

        np.testing.assert_almost_equal(
            orig_arr3x3.numpy(),
            arr3x3.numpy()
        )

        np.testing.assert_almost_equal(
            orig_arr64x64.numpy(),
            arr64x64.numpy()
        )