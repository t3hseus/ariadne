import gin
import torch
import torch.nn as nn

from ariadne.tracknet_v2.model import TrackNETv2
ALLOWED_RNN_TYPES = ['GRU', 'LSTM']


@gin.configurable
class TrackNetClassifier(nn.Module):
    """Builds TrackNETv2_1 classifier model

    # Arguments
        gru_size: number of features in gru output of base model
        coord_size: number of predicted point coords
        num_classes: number of classes to use (real/fake candidate etc)
    """
    def __init__(self, gru_size=32, coord_size=3, num_gru_states=1, z_values=None):
        super().__init__()
        self.gru_feat_block = nn.Sequential(nn.Linear(gru_size*num_gru_states, 30),
                                       #nn.BatchNorm1d(30),
                                       nn.ReLU(),
                                       nn.Linear(30, 15)
                                       )
        self.coord_feat_block = nn.Sequential(
            nn.Linear(coord_size, 20),
            #nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(20, 15)
        )
        self.classifier = nn.Sequential(nn.Linear(30, 20),
                                        #nn.BatchNorm1d(20),
                                        nn.ReLU(),
                                        nn.Linear(20, 1))

    def forward(self, gru_features, coord_features):
        """
        # Arguments
        gru_x: GRU output of base model
        coord_x: coordinates of predicted point (found as last hit) 
        """
        gru_features = gru_features.contiguous().view(gru_features.size()[0], -1)
        x1 = self.gru_feat_block(gru_features.float())
        x2 = self.coord_feat_block(coord_features.float())
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x).float()
        return x