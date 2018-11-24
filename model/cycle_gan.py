import torch.nn.functional as func
import torch.nn as nn

from base import BaseModel

def create_deconvolutional_layer(channels_in, channels_out, kernel_size, stride=2, padding_num=1, use_batch_norm=True):
    """

    :param channels_in:
    :param channels_out:
    :param kernel_size:
    :param stride:
    :param padding_num:
    :param use_batch_norm:
    :return:
    """
    layers = [nn.ConvTranspose2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size,
                                 stride=stride, padding=padding_num, bias=False)]
    if use_batch_norm:
        layers += [nn.BatchNorm2d(num_features=channels_out)]
    return nn.Sequential(*layers)

def create_convolutional_layer(channels_in, channels_out, kernel_size, stride=2, padding_num=1, use_batch_norm=True):
    """

    :param channels_in:
    :param channels_out:
    :param kernel_size:
    :param stride:
    :param padding_num:
    :param use_batch_norm:
    :return:
    """
    layers = [nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding_num, bias=False)]
    if use_batch_norm:
        layers += [nn.BatchNorm2d(num_features=channels_out)]
    return nn.Sequential(*layers)


class Generator(BaseModel):
    """

    """
    def __init__(self, input_class_channels, output_class_channels, internal_channels):
        """

        :param input_class_channels:
        :param output_class_channels:
        :param internal_channels:
        """
        super(Generator, self).__init__()

        self.encoding_layer_one = create_convolutional_layer(input_class_channels, internal_channels, 4)
        self.encoding_layer_two = create_convolutional_layer(internal_channels, internal_channels * 2, 4)

        self.residual_layer_one = create_convolutional_layer(internal_channels * 2, internal_channels * 2, 3, 1, 1)
        self.residual_layer_two = create_convolutional_layer(internal_channels * 2, internal_channels * 2, 3, 1, 1)

        self.decoding_layer_one = create_deconvolutional_layer(internal_channels * 2, internal_channels, 4)
        self.decoding_layer_two = create_deconvolutional_layer(internal_channels, output_class_channels, 4,
                                                               use_batch_norm=False)

    def forward(self, x):
        output = func.leaky_relu(self.encoding_layer_one(x), negative_slope=0.05)
        output = func.leaky_relu(self.encoding_layer_two(output), negative_slope=0.05)

        output = func.leaky_relu(self.residual_layer_one(output), negative_slope=0.05)
        output = func.leaky_relu(self.residual_layer_two(output), negative_slope=0.05)

        output = func.leaky_relu(self.decoding_layer_one(output), negative_slope=0.05)
        output = func.tanh(self.decoding_layer_two(output))

        return output


class Discriminator(BaseModel):
    """

    """
    def __init__(self, input_class_channels, n_output_labels, internal_channels):
        """

        :param input_class_channels:
        :param n_output_labels:
        :param internal_channels:
        """
        super(Discriminator, self).__init__()

        self.conv_layer_one = create_convolutional_layer(input_class_channels, internal_channels, 4,
                                                         use_batch_norm=False)
        self.conv_layer_two = create_convolutional_layer(internal_channels, internal_channels * 2, 4)
        self.conv_layer_three = create_convolutional_layer(internal_channels * 2, internal_channels * 4, 4)
        self.conv_layer_four = create_convolutional_layer(internal_channels * 4, n_output_labels, 4, 1, 0,
                                                          use_batch_norm=False)

    def forward(self, x):
        out = func.leaky_relu(self.conv_layer_one(x), negative_slope=0.05)
        out = func.leaky_relu(self.conv_layer_two(out), negative_slope=0.05)
        out = func.leaky_relu(self.conv_layer_three(out), negative_slope=0.05)
        out = self.conv_layer_four(out).squeeze()
        return out
