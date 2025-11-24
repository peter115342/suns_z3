import torch
import torch.nn as nn


class IrradianceCNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        conv_channels=[32, 64, 128],
        fc_hidden=[512, 256],
        dropout_rate=0.5,
        use_batch_norm=True,
    ):
        super(IrradianceCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for out_channels in conv_channels:
            layers = []
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout_rate * 0.5))

            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        fc_input_size = conv_channels[-1] * 7 * 7

        self.fc_layers = nn.ModuleList()
        prev_size = fc_input_size

        for hidden_size in fc_hidden:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            self.fc_layers.append(nn.ReLU(inplace=True))
            self.fc_layers.append(nn.Dropout(p=dropout_rate))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = self.output_layer(x)

        return x.squeeze(1)


def create_cnn_model(
    conv_channels=[32, 64, 128],
    fc_hidden=[512, 256],
    dropout_rate=0.5,
    use_batch_norm=True,
    device=None,
):
    if device is None:
        from .utils import get_device

        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    model = IrradianceCNN(
        input_channels=3,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )

    model = model.to(device)

    return model

