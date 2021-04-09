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
    def __init__(self, gru_size=32, coord_size=2, num_classes=2):
        super().__init__()
        self.gru_feat_block = nn.Sequential(nn.Linear(gru_size*2, 30),
                                       #nn.BatchNorm1d(30),
                                       nn.ReLU(),
                                       nn.Linear(30, 30)
                                       )
        self.coord_feat_block = nn.Sequential(
            nn.Linear(coord_size, 30),
            #nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 30)
        )
        self.classifier = nn.Sequential(nn.Linear(60, 30),
                                        #nn.BatchNorm1d(20),
                                        nn.ReLU(),
                                        nn.Linear(30, 1))

    def forward(self, gru_features, coord_features):
        """
        # Arguments
        gru_x: GRU output of base model
        coord_x: coordinates of predicted point (found as last hit) 
        """
        # BxTxC -> BxCxT
        gru_features = gru_features.contiguous().view(gru_features.size()[0], -1)
        x1 = self.gru_feat_block(gru_features.float())
        x2 = self.coord_feat_block(coord_features.float())
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x).float()
        return x



@gin.configurable
class TrackNETv2_1(nn.Module):
    """Builds TrackNETv2 model

    # Arguments
        input_features: number of input features (channels)
        rnn_type: type of the rnn unit, one of [`lstm`, `gru`]
        conv_features: number of convolutional channels
        batch_first: if True, size of inputs is BxTxC
    """
    def __init__(self,
                 input_features=4,
                 conv_features=32,
                 rnn_type='gru',
                 batch_first=True,
                 gru_size=16,
                 coord_size=2):
        super().__init__()
        self.base_model = TrackNETv2(input_features=input_features, conv_features=conv_features, rnn_type=rnn_type, batch_first=batch_first)
        self.classifier = TrackNetClassifier(gru_size=16, coord_size=coord_size)

    def forward(self, inputs, input_lengths):
        class_dict = self.base_model.get_tracknet_v2_1_inputs(inputs, input_lengths)
        gru_x = class_dict['last_layer']
        tracknet_x = class_dict['model_outputs']
        x = self.classifier(gru_x, tracknet_x)
        return x

