import torch
from torch import nn
import torch.nn.functional as f


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, input_batch):
        block_out = torch.relu_(self.conv1(input_batch))
        block_out = torch.relu_(self.conv2(block_out))

        return f.max_pool3d(block_out, 2, 2)
