import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_size, output_shape):
        super().__init__()
        self.output_shape = output_shape

        self.layers = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape)
        )

    def forward(self, x):
        return self.layers(x)
