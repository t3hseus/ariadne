import gin
import torch
import torch.nn as nn

from ariadne.tracknet_v2.model import TrackNETv2
ALLOWED_RNN_TYPES = ['GRU', 'LSTM']


@gin.configurable
class TrackNetV22Classifier(nn.Module):
    """Builds TrackNETv2_2 classifier model

    # Arguments
        gru_size: number of features in gru output of base model
        coord_size: number of predicted point coords
        num_classes: number of classes to use (real/fake candidate etc)
    """
    def __init__(self, gru_size=32, coord_size=2, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(9, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, coord_features):
        """
        # Arguments
        gru_x: GRU output of base model
        coord_x: coordinates of predicted point (found as last hit) 
        """
        # BxTxC -> BxCxT
        x = self.classifier(coord_features).float()
        return x

@gin.configurable
class TrackNetV22ClassifierBig(nn.Module):
    """Builds TrackNETv2_2 classifier model

    # Arguments
        gru_size: number of features in gru output of base model
        coord_size: number of predicted point coords
        num_classes: number of classes to use (real/fake candidate etc)
    """
    def __init__(self, gru_size=32, coord_size=2, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(9, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )

    def forward(self, coord_features):
        """
        # Arguments
        gru_x: GRU output of base model
        coord_x: coordinates of predicted point (found as last hit)
        """
        # BxTxC -> BxCxT
        x = self.classifier(coord_features).float()
        return x