import gin

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


LOGGER = logging.getLogger('ariadne.pointnet')


def _build_global_pooling_layer(pooling):
    # pool and flatten
    if pooling == 'max':
        global_pooling = nn.AdaptiveMaxPool1d(1)
    elif pooling == 'mean' or pooling == 'average':
        global_pooling = nn.AdaptiveAvgPool1d(1)
    else:
        raise ValueError(f'Pooling type `{pooling}` is not supported! '
                            'Chose one from `mean` or `max`')
    return nn.Sequential(
        global_pooling,
        nn.Flatten()
    )

@gin.configurable
class TNet(nn.Module):
    def __init__(self,
                k=3,
                conv_layers=(8, 16),
                linear_layers=None,
                pooling='max'):
        super().__init__()
        self.k = k
        if linear_layers is None:
            linear_layers = tuple(reversed(conv_layers))

        features_layers = []

        # convolution blocks
        in_features = self.k
        for i, out_features in enumerate(conv_layers):
            conv_block = nn.Sequential(
                nn.Conv1d(in_features, out_features, 1),
                nn.BatchNorm1d(out_features),
                nn.PReLU()
            )
            features_layers.append((f'conv{i+1}', conv_block))
            in_features = out_features

        # pool and flatten
        features_layers.append((
            'global_pool', _build_global_pooling_layer(pooling)
        ))

        # linear blocks
        for i, out_features in enumerate(linear_layers):
            linear_block = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.PReLU()
            )
            features_layers.append((f'fc{i+1}', linear_block))
            in_features = out_features

        self.features = nn.Sequential(OrderedDict(features_layers))

        # transform block
        self.transform = nn.Linear(in_features, k*k)


    def forward(self, input):
        x = self.features(input)
        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(input.size(0), 1, 1)
        init = init.to(x.device)
        return self.transform(x).view(-1,self.k,self.k) + init

@gin.configurable
class Transform(nn.Module):
    def __init__(self,
                 input_features_dim=3,
                 input_transform=None,
                 feature_transform=None,
                 add_top_relu=False,
                 top_layers=(16, 32),
                 pooling='max'):
        super().__init__()
        assert input_transform is None or isinstance(input_transform, TNet)
        assert feature_transform is None or isinstance(feature_transform, TNet)
        assert len(top_layers) >= 1
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        self.output_size = top_layers[-1]

        if input_transform is not None:
            LOGGER.warning(
                f'The value `input_features_dim={input_features_dim}` '
                f'will be overwritten by the parameter `k={input_transform.k}` '
                'of the input_transform argument'
            )
            input_features_dim = input_transform.k
            self.input_transform = input_transform

        in_features = input_features_dim
        if self.feature_transform is not None:
            self.input2features = nn.Sequential(
                nn.Conv1d(in_features, self.feature_transform.k, 1),
                nn.BatchNorm1d(self.feature_transform.k),
                nn.PReLU()
            )
            in_features = self.feature_transform.k
            #self.output_size += self.feature_transform.k
        else:
            pass
            # calculate the size of the output
            # cat([local_features, global_features])
            #self.output_size += input_features_dim

        self.top_layers = []
        for i, out_features in enumerate(top_layers):
            conv_block = nn.Sequential(
                nn.Conv1d(in_features, out_features, 1),
                nn.BatchNorm1d(out_features),
                nn.PReLU()
            )
            if i == len(top_layers) - 1 and not add_top_relu:
                # without last PReLU
                self.top_layers.append((f'conv{i+1}', conv_block[:-1]))
            else:
                self.top_layers.append((f'conv{i+1}', conv_block))
            in_features = out_features

        self.top_layers.append((
            'global_pool', _build_global_pooling_layer(pooling)
        ))
        # create a Sequential block
        self.top_layers = nn.Sequential(OrderedDict(self.top_layers))


    def _transform_features(self, input, transform):
        transform_matrix = transform(input)
        output = torch.bmm(
            torch.transpose(input, 1, 2),
            transform_matrix
        ).transpose(1, 2)
        return output#, transform_matrix


    def forward(self, input):
        """
        Returns:
            1) [local features, global features],
            2) input_transform_matrix,
            3) features_transform_matrix
        """
        input_transform_mtrx = None
        feature_transform_mtrx = None
        x = input
        local_features = input

        if self.input_transform is not None:
            x = self._transform_features(x, self.input_transform)
            x = self.input2features(x)
            local_features = x

        if self.feature_transform is not None:
            x = self._transform_features(x, self.feature_transform)
            local_features = x

        global_features = self.top_layers(x)
        # repeat global features for each local feature
        #global_features = global_features.repeat(local_features.size(-1), 1, 1).permute(1, 2, 0)
        #output = torch.cat([local_features, global_features], 1)
        return global_features#, input_transform_mtrx, feature_transform_mtrx

@gin.configurable
class PointNet(nn.Module):
    def __init__(self,
                 transform,
                 fc_layers=(128, 64),
                 classes=2,
                 softmax_for_binary=False):
        super().__init__()
        assert transform is None or isinstance(transform, Transform)
        assert classes >= 2
        self.transform = transform
        # build graph
        features_layers = []
        in_features = self.transform.output_size

        # linear blocks
        for i, out_features in enumerate(fc_layers):
            linear_block = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.PReLU()
            )
            features_layers.append((f'fc{i + 1}', linear_block))
            in_features = out_features

        self.features = nn.Sequential(OrderedDict(features_layers))
        # # last classification layer
        # if classes == 2 and not softmax_for_binary:
        #     # sigmoid + binary cross-entropy
        #     self.classifier = nn.Conv1d(in_features, 1, 1)
        # else:
        #     raise NotImplementedError # todo for multi classes remove output squeeze
        #     # softmax + cross-entropy
        #     self.classifier = nn.Conv1d(in_features, classes, 1)

    def forward(self, x):
        #x, input_transform_mtrx, feature_transform_mtrx = self.transform(x)
        x = self.transform(x)
        x = self.features(x)
        #x = self.top_layers(x)
        #output = self.classifier(x)
        # (bsz, classes, n_points) -> (bsz, n_points, classes)
        #output = output.transpose(2,1).contiguous()

        return [x]#, input_transform_mtrx, feature_transform_mtrx

if __name__ == '__main__':
    # remove this after merging
    # python ariadne/point_net_dev/model.py
    # pytest ariadne/point_net_dev/test_model.py
    from loss import PointNetClassificationLoss
    in_net = TNet(3)
    feat_net = TNet(32)
    trans = Transform(input_transform=in_net, feature_transform=feat_net)
    point_net = PointNet(trans)
    print(point_net)
    print(f'Number of parameters: {sum(p.numel() for p in point_net.parameters())}\n')
    preds = point_net(torch.rand((256, 3, 5000)))
    # loss
    loss = PointNetClassificationLoss(alpha=0.1) # leave the default alpha, this only for experiment
    labels = torch.cat([torch.zeros(128, 5000), torch.ones(128, 5000)])
    labels = torch.unsqueeze(labels, 2)
    loss_val = loss(preds, labels)
    # None transform matrices
    print(loss((preds[0], None, None), labels))
    print(loss((preds[0], preds[1], None), labels))
    print(loss((preds[0], None, preds[2]), labels))
    # manual with squeezing
    # Compute target weights on-the-fly for loss function
    batch_weights_real = labels * loss.real_weight
    batch_weights_fake = (1 - labels) * loss.fake_weight
    batch_weights = batch_weights_real + batch_weights_fake
    # unpack predictions
    preds, in_mtrx, feat_mtrx = preds
    preds, labels, weight = torch.squeeze(preds, 2), torch.squeeze(labels, 2), torch.squeeze(batch_weights, 2)
    print(preds.shape, labels.shape, loss.criterion, batch_weights.shape)
    ce_loss_val = loss.criterion(preds, labels, weight=weight)
    reg_loss = 0
    reg_loss += loss._regularization_term(in_mtrx)
    reg_loss += loss._regularization_term(feat_mtrx)
    reg_loss = loss.alpha * reg_loss / float(preds.size(0))
    manual_loss_val = ce_loss_val + reg_loss
    print(manual_loss_val, loss_val)