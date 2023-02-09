import torch
from torch import nn
from numpy import pi

class SelfAttentionBlock(nn.Module):
        """Self attention block which learns to identify and highlight
        features in the input vector which are strongly related. As
        described in Vaswani+2017:
        https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
        
        Args:
            channels (int): channel dimension for input vector x
            init_var_out (float): variance for initialization of output attention layer. Set << 1 to start training with near skip-connection.

        """

    def __init__(self, channels, init_var_out = 1e-2, *args, **kwargs):
        super().__init__()

        self.query = nn.Linear(
            in_features = channels,
            out_features = channels,
        )
        self.key = nn.Linear(
            in_features = channels,
            out_features = channels,
        )
        self.value = nn.Linear(
            in_features = channels,
            out_features = channels,
        )
        self.out = nn.Linear(
            in_features = channels,
            out_features = channels,
        )

        with torch.no_grad():
            std = (init_var / channels) ** 0.5
            self.query.weight.data.normal_(0., (1. / channels) ** 0.5)
            self.key.weight.data.normal_(0., (1. / channels) ** 0.5)
            self.value.weight.data.normal_(0., (1. / channels) ** 0.5)
            self.out.weight.data.normal_(0., (init_var / channels) ** 0.5)
            self.query.bias.zero_()
            self.key.bias.zero_()
            self.value.bias.zero_()
            self.out.bias.zero_()

    def forward(self, x):

        # fixme dimension alignment permute/view
        querys, keys, values = self.query(x), self.key(x), self.value(x)
        
        weight = torch.bmm(querys, keys)

        softmax_weight = torch.softmax(weight, dim = -1)

        attention = torch.bmm(weight, values)

        return self.out(attention) + x

class FourierProjection(nn.Module):
    """Module for projecting the time component into a vector embedded in
    sinusoid functions. This feature is more learnable for the network
    than a direct time float value.

    """
    def __init__(self, embedding_dimensions, scale = 30., *args, **kwargs):
        super().__init__()

        assert embedding_dimensions % 2

        self.W = torch.randn(embedding_dimensions // 2) * scale

    def forward(self, t):

        t_projection = t.view(-1, 1) * self.W.view(1, -1) * 2 * pi

        t_embedding = torch.cat(
            (torch.sin(t_projection), torch.cos(t_projection)),
            dim = -1
        )

        return t_embedding

class ScalingLayer(nn.Module):
    """Layer which adjusts the size of the input tensor by some
    scale. This fills the role of a Downsampling Layer or an
    Upsampling Layer.

    Args:
      scale_direction (str): Direction in which to scale the data. Options: UP, DOWN
      in_channels (int): Number of channels for input tensor
      scale_factor (int): degree by which to scale the input.
      scale_method (str): Technique used to do the scaling. Defaults to max pooling for DOWN, and Conv2DTranspose for UP
      kernel_size (tuple of int): Kernel used during size scaling

    """
    
    def __init__(
            self,
            scale_direction,
            in_channels = None,
            out_channels = None,
            scale_factor = 2,
            scale_method = None,
            kernel_size = (3, 3),
    ):

        if scale_direction == "DOWN":
            self.scaler = nn.MaxPooling2D(
                kernel_size = (scale_factor, scale_factor),
            )
        elif scale_direction == "UP":
            out_channels = in_channels / scale_factor if out_channels is None else out_channels
            self.scaler = nn.Conv2DTranspose(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = (scale_factor, scale_factor),
            )
        else:
            raise ValueError("scale_direction should be one of: UP, DOWN")
        
    def forward(self, x):
        
        scaled_input = self.scaler(x)
        return scaled_input

class ResidualBlock_Concept(nn.Module):
    """Implements the conceptual core of a residual block, essentially
    that there is a skip connection over the block.

    Args:
      internal_block (nn.Module): A module for the neural network which has output of the same dimension as the input.

    """
    def __init__(self, internal_block):

        self.internal_block = internal_block

    def forward(self, x):

        base_block = self.internal_block(x)

        return base_block + x

class ResidualBlock_He2015(ResidualBlock_Concept):
    """Residual block as implemented in He+2015:
    https://arxiv.org/pdf/1512.03385.pdf

    """
    def __init__(self, in_channels, out_channels, stride = 1):
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU()

        super().__init__(self.block)

    def forward(self, x):

        resblock = super().forward(x)

        activation = self.relu(resblock)
        
        return activation
