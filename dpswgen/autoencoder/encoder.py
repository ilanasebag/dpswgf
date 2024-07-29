import torch.nn as nn

from autoencoder.conv import DCGAN32Decoder, DCGAN32Encoder, DCGAN64Decoder, DCGAN64Encoder, LiutkusDecoder, LiutkusEncoder, VGG64Decoder, VGG64Encoder
from autoencoder.mlp import MLP


class NormalizedEncoder(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return nn.functional.normalize(self.network(x))


def encoder_factory(input_shape, out_size, config):
    name = config.name
    if name == 'mlp':
        return MLP(input_shape, out_size, config.width, config.depth, config.activation)
    assert len(input_shape) == 3
    if name == 'liutkus':
        return LiutkusEncoder(input_shape, out_size)
    if name == 'dcgan':
        assert input_shape[1] == input_shape[2] and input_shape[2] in [32, 64]
        if input_shape[2] == 32:
            return DCGAN32Encoder(input_shape[0], out_size, config.width)
        else:
            return DCGAN64Encoder(input_shape[0], out_size, config.width)
    if name == 'vgg':
        assert input_shape[1] == input_shape[2] == 64
        return VGG64Encoder(input_shape[0], out_size, config.width)
    raise ValueError(f'No encoder named \'{name}\'')


def normalized_encoder_factory(input_shape, out_size, config):
    return NormalizedEncoder(encoder_factory(input_shape, out_size, config))


def decoder_factory(in_size, output_shape, config):
    name = config.name
    if name == 'mlp':
        return MLP(in_size, output_shape, config.width, config.depth, config.activation)
    assert len(output_shape) == 3
    if name == 'liutkus':
        return LiutkusDecoder(in_size, output_shape)
    if name == 'dcgan':
        assert output_shape[1] == output_shape[2] and output_shape[2] in [32, 64]
        if output_shape[2] == 32:
            return DCGAN32Decoder(output_shape[0], in_size, config.width, False)
        else:
            return DCGAN64Decoder(output_shape[0], in_size, config.width, False)
    if name == 'vgg':
        assert output_shape[1] == output_shape[2] == 64
        return VGG64Decoder(output_shape[0], in_size, config.width, False)
    raise ValueError(f'No encoder named \'{name}\'')
