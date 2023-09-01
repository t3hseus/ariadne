import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from performer_pytorch import Performer
from transformers import PerceiverConfig
from transformers.models.perceiver import PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    AbstractPreprocessor,
    PerceiverBasicDecoder,
    PerceiverProjectionPostprocessor,
)


class PerceiverCloudPreprocessor(AbstractPreprocessor):
    """
    Hits preprocessing for Perceiver Encoder.
    Can be used to add positional encodings to input.

    The dimensionality of the embeddings is determined by
    the `d_model` attribute of the configuration.

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config = config
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )

    @property
    def num_channels(self) -> int:
        return self.config.d_model

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """inputs: [B, N, C]
        returns: [B, N, C], None, None - to meet HF interface
        """
        seq_length = inputs.shape[1]
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        embeddings = inputs + self.position_embeddings(position_ids)

        return embeddings, None, None


class Preprocessor(AbstractPreprocessor):
    """
    Input Embedding layer which consist of 2 stacked linear
    layers with batchnorms and relu.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.conv1 = nn.Conv1d(3, self.config.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(self.config.d_model, self.config.d_model, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(self.config.d_model)
        self.bn2 = nn.BatchNorm1d(self.config.d_model)

    @property
    def num_channels(self) -> int:
        return self.config.d_model

    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]

        Output
            x: [B, out_channels, N], None, None - to neet HF interface
        """

        x = F.relu(self.conv1(x))  # x.transpose(1, -1)
        x = F.relu(self.conv2(x))
        return x.transpose(1, -1), None, None


@gin.configurable
class HFPerceiver(nn.Module):
    """Perceiver model based on HF Perceiver."""

    def __init__(self, d_model=32, num_heads=2, d_latents=24, num_latents=1024):
        super().__init__()

        config = PerceiverConfig(
            d_model=d_model,
            num_heads=num_heads,
            d_latents=d_latents,
            num_latents=num_latents,
            num_cross_attention_heads=num_heads,
            num_self_attention_heads=num_heads,
            max_position_embeddings=num_latents,
        )
        self.config = config
        preprocessor = Preprocessor(config)
        decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=config.d_latents,
            num_channels=config.d_latents,
            trainable_position_encoding_kwargs=dict(
                num_channels=config.d_latents, index_dims=512
            ),
            use_query_residual=True,
        )
        self.model = PerceiverModel(
            config=config,
            input_preprocessor=preprocessor,
            decoder=decoder,
            output_postprocessor=PerceiverProjectionPostprocessor(config.d_latents, 1),
        )
        self.postprocessor = nn.Sequential(
            nn.Conv1d(config.d_latents, int(config.d_latents / 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(int(config.d_latents / 2), 1, kernel_size=1),
        )

        self.linear = nn.Linear(config.d_latents, 1)

        # self.postprocessor = nn.Sequential(
        #     nn.Linear(4, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        # )

    def forward(self, **inputs):
        outputs = self.model(
            inputs["x"],
            # attention_mask=inputs['mask'],
            return_dict=True,
            output_attentions=True,
        )
        # outputs = inputs['x']
        # return self.postprocessor(outputs.last_hidden_state)
        # self.postprocessor(outputs.logits.transpose(-1, 1)).transpose(-1, 1)
        return outputs.logits  # self.linear(outputs.logits)


def get_perceiver_decoder(config):
    """This function is not used and is saved only to save
    memory of that is configured for decoder"""
    trainable_position_encoding_kwargs_decoder = dict(
        num_channels=3, index_dims=config.max_position_embeddings
    )
    num_fourier_bands = 3
    fourier_position_encoding_kwargs_decoder = dict(
        num_bands=num_fourier_bands, max_resolution=[config.max_position_embeddings]
    )

    decoder = PerceiverBasicDecoder(
        config,
        num_channels=3,
        output_num_channels=1,
        output_index_dims=1,  # config.max_position_embeddings,
        qk_channels=3 * 8,
        v_channels=3,
        num_heads=3,
        use_query_residual=False,
        position_encoding_type="trainable",
        final_project=True,
        trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
        fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
    )
    return decoder


@gin.configurable
class HFPerformer(nn.Module):
    """Performer model based on HF Perceiver."""

    def __init__(self, d_model=32, num_heads=2, d_latents=24, num_latents=1024):
        super().__init__()

        config = PerceiverConfig(
            d_model=d_model,
            num_heads=num_heads,
            d_latents=d_latents,
            num_latents=num_latents,
            num_cross_attention_heads=num_heads,
            num_self_attention_heads=num_heads,
            max_position_embeddings=num_latents,
        )
        self.config = config
        preprocessor = Preprocessor(config)
        decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=config.d_latents,
            num_channels=config.d_latents,
            trainable_position_encoding_kwargs=dict(
                num_channels=config.d_latents, index_dims=512
            ),
            use_query_residual=True,
        )
        self.model = PerceiverModel(
            config=config,
            input_preprocessor=preprocessor,
            decoder=decoder,
            output_postprocessor=PerceiverProjectionPostprocessor(config.d_latents, 1),
        )
        self.postprocessor = nn.Sequential(
            nn.Conv1d(config.d_latents, int(config.d_latents / 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(int(config.d_latents / 2), 1, kernel_size=1),
        )

        self.linear = nn.Linear(config.d_latents, 1)

        # self.postprocessor = nn.Sequential(
        #     nn.Linear(4, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        # )

    def forward(self, **inputs):
        outputs = self.model(
            inputs["x"],
            # attention_mask=inputs['mask'],
            return_dict=True,
            output_attentions=True,
        )
        print(outputs.logits)
        # outputs = inputs['x']
        # return self.postprocessor(outputs.last_hidden_state)
        # self.postprocessor(outputs.logits.transpose(-1, 1)).transpose(-1, 1)
        return outputs.logits  # self.linear(outputs.logits)


@gin.configurable
class Performer(nn.Module):
    def __init__(self, n_feat=3):
        super().__init__()
        # hugginface perceiver
        # attention mask to padded hits +1
        # loss with sigmoid in it +1
        # not use perceiverdecoder etc., need small model and another dimensions
        # maybe 2 neurons with softmax - and crossentropy
        self.model = Performer(dim=n_feat, depth=6, heads=3, causal=False, dim_head=128)

        self.decoder = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        x = self.decoder(x)
        return x


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)