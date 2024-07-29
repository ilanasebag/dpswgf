import torch
import torch.nn as nn

from DPSWflow.autoencoder.utils import activation_factory


#Code from https://github.com/aliutkus/swf
# no known license, however this code has been used and is being used here for research purpose only 

class LiutkusEncoder(nn.Module):
    def __init__(self, input_shape, out_size):
        super().__init__()
        self.input_shape = input_shape
        conv1 = nn.Conv2d(self.input_shape[0], 3, 3, stride=1, padding=1)
        conv2 = nn.Conv2d(3, input_shape[-1], 2, stride=2, padding=0)
        conv3 = nn.Conv2d(input_shape[-1], input_shape[-1], 3, stride=1, padding=1)
        conv4 = nn.Conv2d(input_shape[-1], input_shape[-1], 3, stride=1, padding=1)
        fc1 = nn.Linear(int(input_shape[-1] ** 3 / 4), out_size)
        relu = nn.ReLU(inplace=True)
        flatten = nn.Flatten()
        self.network = nn.Sequential(conv1, relu, conv2, relu, conv3, relu, conv4, relu, flatten, fc1)

    def forward(self, x):
        return self.network(x)


class LiutkusDecoder(nn.Module):
    def __init__(self, in_size, output_shape):
        super().__init__()
        self.output_shape = output_shape
        d = output_shape[-1]
        self.fc4 = nn.Linear(in_size, int(d / 2 * d / 2 * d))
        deconv1 = nn.ConvTranspose2d(d, d, 3, stride=1, padding=1)
        deconv2 = nn.ConvTranspose2d(d, d, 3, stride=1, padding=1)
        deconv3 = nn.ConvTranspose2d(d, d, 2, stride=2, padding=0)
        conv5 = nn.Conv2d(d, self.output_shape[0], 3, stride=1, padding=1)
        relu = nn.ReLU(inplace=True)
        sigmoid = nn.Sigmoid()
        self.conv_network = nn.Sequential(deconv1, relu, deconv2, relu, deconv3, relu, conv5, sigmoid)

    def forward(self, x):
        d = self.output_shape[-1]
        out = torch.relu(self.fc4(x))
        out = out.view(-1, d, int(d / 2), int(d / 2))
        return self.conv_network(out)


#the following code uses the following lisence
# Copyright 2020 Anonymous Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def make_conv_block(conv, activation, bn=True):
    """
    Supplements a convolutional block with activation functions and batch normalization.

    Parameters
    ----------
    conv : torch.nn.Module
        Convolutional block.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function, or no
        activation if 'none' is chosen, after the convolution.
    bn : bool
        Whether to add batch normalization after the activation.

    Returns
    -------
    torch.nn.Sequential
        Sequence of the input convolutional block, the potentially chosen activation function, and the potential batch
        normalization.
    """
    out_channels = conv.out_channels
    modules = [conv]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if activation != 'none':
        modules.append(activation_factory(activation))
    return nn.Sequential(*modules)


class BaseEncoder(nn.Module):
    """
    Module implementing the encoders forward method.

    Attributes
    ----------
    nh : int
        Number of dimensions of the output flat vector.
    """
    def __init__(self, nh):
        """
        Parameters
        ----------
        nh : int
            Number of dimensions of the output flat vector.
        """
        super().__init__()
        self.nh = nh

    def forward(self, x, return_skip=False):
        """
        Parameters
        ----------
        x : torch.*.Tensor
            Encoder input.
        return_skip : bool
            Whether to extract and return, besides the network output, skip connections.

        Returns
        -------
        torch.*.Tensor
            Encoder output as a tensor of shape (batch, size).
        list
            Only if return_skip is True. List of skip connections represented as torch.*.Tensor corresponding to each
            convolutional block in reverse order (from the deepest to the shallowest convolutional block).
        """
        skips = []
        h = x
        for layer in self.conv:
            h = layer(h)
            skips.append(h)
        h = self.last_conv(h).view(-1, self.nh)
        if return_skip:
            return h, skips[::-1]
        return h


class DCGAN32Encoder(BaseEncoder):
    """
    Module implementing the DCGAN encoder.
    """
    def __init__(self, nc, nh, nf):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super().__init__(nh)
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(nc, nf, 3, 1, 1, bias=False), activation='leaky_relu', bn=False),
            make_conv_block(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False), activation='leaky_relu')
        ])
        self.last_conv = make_conv_block(nn.Conv2d(nf * 8, nh, 4, 1, 0, bias=False), activation='tanh')


class DCGAN64Encoder(BaseEncoder):
    """
    Module implementing the DCGAN encoder.
    """
    def __init__(self, nc, nh, nf):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super().__init__(nh)
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(nc, nf, 4, 2, 1, bias=False), activation='leaky_relu', bn=False),
            make_conv_block(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False), activation='leaky_relu')
        ])
        self.last_conv = make_conv_block(nn.Conv2d(nf * 8, nh, 4, 1, 0, bias=False), activation='tanh')


class VGG64Encoder(BaseEncoder):
    """
    Module implementing the VGG encoder.
    """
    def __init__(self, nc, nh, nf):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super().__init__(nh)
        self.conv = nn.ModuleList([
            nn.Sequential(
                make_conv_block(nn.Conv2d(nc, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), activation='leaky_relu'),
            )
        ])
        self.last_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            make_conv_block(nn.Conv2d(nf * 8, nh, 4, 1, 0, bias=False), activation='tanh')
        )


class BaseDecoder(nn.Module):
    """
    Module implementing the decoders forward method.

    Attributes
    ----------
    ny : int
        Number of dimensions of the output flat vector.
    skip : bool
        Whether to include skip connections into the decoder.
    """
    def __init__(self, ny, skip):
        """
        Parameters
        ----------
        ny : int
            Number of dimensions of the input flat vector.
        """
        super().__init__()
        self.ny = ny
        self.skip = skip
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, skip=None):
        """
        Parameters
        ----------
        z : torch.*.Tensor
            Decoder input.
        skip : list
            List of torch.*.Tensor representing skip connections in the same order as the decoder convolutional
            blocks. Must be None when skip connections are not allowed.
        sigmoid : bool
            Whether to apply a sigmoid at the end of the decoder.

        Returns
        -------
        torch.*.Tensor
            Decoder output as a frame of shape (batch, channels, width, height).
        """
        assert skip is None and not self.skip or self.skip and skip is not None
        h = self.first_upconv(z.view(*z.shape, 1, 1))
        for i, layer in enumerate(self.conv):
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h = layer(h)
        x_ = h
        return self.sigmoid(x_)


class DCGAN32Decoder(BaseDecoder):
    """
    Module implementing the DCGAN decoder.
    """
    def __init__(self, nc, ny, nf, skip):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder architecture.
        skip : list
            List of torch.*.Tensor representing skip connections in the same order as the decoder convolutional
            blocks. Must be None when skip connections are not allowed.
        """
        super().__init__(ny, skip)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = make_conv_block(nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0, bias=False), activation='leaky_relu')
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(nf * 8 * coef, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.ConvTranspose2d(nf * 4 * coef, nf * 2, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.ConvTranspose2d(nf * 2 * coef, nf, 4, 2, 1, bias=False), activation='leaky_relu'),
            nn.ConvTranspose2d(nf * coef, nc, 4, 2, 1, bias=False),
        ])


class DCGAN64Decoder(BaseDecoder):
    """
    Module implementing the DCGAN decoder.
    """
    def __init__(self, nc, ny, nf, skip):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder architecture.
        skip : list
            List of torch.*.Tensor representing skip connections in the same order as the decoder convolutional
            blocks. Must be None when skip connections are not allowed.
        """
        super().__init__(ny, skip)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = make_conv_block(nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0, bias=False), activation='leaky_relu')
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(nf * 8 * coef, nf * 4, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.ConvTranspose2d(nf * 4 * coef, nf * 2, 4, 2, 1, bias=False), activation='leaky_relu'),
            make_conv_block(nn.ConvTranspose2d(nf * 2 * coef, nf, 4, 2, 1, bias=False), activation='leaky_relu'),
            nn.ConvTranspose2d(nf * coef, nc, 4, 2, 1, bias=False),
        ])


class VGG64Decoder(BaseDecoder):
    """
    Module implementing the VGG decoder.
    """
    def __init__(self, nc, ny, nf, skip):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder architecture.
        skip : list
            List of torch.*.Tensor representing skip connections in the same order as the decoder convolutional
            blocks. Must be None when skip connections are not allowed.
        """
        super().__init__(ny, skip)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = nn.Sequential(
            make_conv_block(nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0, bias=False), activation='leaky_relu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 8 * coef, nf * 8, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 4 * coef, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 2 * coef, nf * 2, 3, 1, 1, bias=False), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * coef, nf, 3, 1, 1, bias=False), activation='leaky_relu'),
                nn.ConvTranspose2d(nf, nc, 3, 1, 1, bias=False),
            ),
        ])
