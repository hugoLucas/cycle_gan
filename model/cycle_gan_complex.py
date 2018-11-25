import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, n_input_channels, n_frames=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()

        model = [
            nn.Conv2d(n_input_channels, n_frames, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        n_frames_current, n_frames_previous = 1, 1
        for ii in range(1, n_layers):
            n_frames_previous, n_frames_current = n_frames_current, min(2 ** ii, 8)
            model += [
                nn.Conv2d(n_frames * n_frames_previous, n_frames * n_frames_current, kernel_size=4, stride=2, padding=1,
                          bias=True),
                nn.InstanceNorm2d(n_frames * n_frames_current),
                nn.LeakyReLU(0.2, True)
            ]

        n_frames_previous, n_frames_current = n_frames_current, min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(n_frames * n_frames_previous, n_frames * n_frames_current, kernel_size=4, stride=1, padding=1,
                      bias=True),
            nn.InstanceNorm2d(n_frames * n_frames_current),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_frames * n_frames_current, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetGenerator(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, n_frames=64, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_input_channels, n_frames, kernel_size=7, padding=0),
            nn.InstanceNorm2d(n_frames),
            nn.ReLU(True)]

        n_down_sample_layers = 2
        for ii in range(n_down_sample_layers):
            layer_channels = (2 ** ii) * n_frames
            model += [
                nn.Conv2d(layer_channels, layer_channels * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(layer_channels * 2),
                nn.ReLU(True)
            ]

        layer_channels = 2 ** n_down_sample_layers
        for ii in range(n_blocks):
            model.append(ResnetBlock(n_frames * layer_channels))

        for ii in range(n_down_sample_layers):
            layer_channels = n_frames * (2 ** (n_down_sample_layers - ii))
            model += [nn.ConvTranspose2d(layer_channels, int(layer_channels/2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=True),
                      nn.InstanceNorm2d(int(layer_channels/2)),
                      nn.ReLU(True)]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_frames, n_output_channels, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, n_input_channels):
        super(ResnetBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_input_channels, n_input_channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_input_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_input_channels, n_input_channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(n_input_channels),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
