import torch.nn as nn


def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.

    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.

    Returns
    -------
    torch.nn.Module
        Element-wise activation layer.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Activation function \'{name}\' not yet implemented')
