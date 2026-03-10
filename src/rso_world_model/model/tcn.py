from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.output_activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.output_activation(self.net(x) + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        for index, out_channels in enumerate(channels):
            dilation = 2**index
            in_channels = num_inputs if index == 0 else channels[index - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

