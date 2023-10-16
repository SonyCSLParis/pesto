from functools import partial

import torch
import torch.nn as nn


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features+out_features-1,
            padding=out_features-1,
            bias=False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class PESTOEncoder(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report,
    but for longer HCQT input (always stride 1 in time)
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).
    Outputs one channel with sigmoid activation.

    Args (Defaults: BasicCNN by Johannes Zeitler but with 1 input channel):
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        residual:         If True, use residual connections for prefiltering (default: False)
        n_bins_in:        Number of input bins (12 * number of octaves)
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch, num_octaves * 12)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """

    def __init__(
            self,
            n_chan_layers=(20, 20, 10, 1),
            n_prefilt_layers=1,
            residual=False,
            n_bins_in=216,
            output_dim=128,
            num_output_layers: int = 1
    ):
        super(PESTOEncoder, self).__init__()

        activation_layer = partial(nn.LeakyReLU, negative_slope=0.3)

        n_ch = n_chan_layers
        if len(n_ch) < 5:
            n_ch.append(1)

        # Layer normalization over frequency
        self.layernorm = nn.LayerNorm(normalized_shape=[1, n_bins_in])

        # Prefiltering
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_ch[0], kernel_size=15, padding=7, stride=1),
            activation_layer()
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_list = nn.ModuleList()
        for p in range(1, n_prefilt_layers):
            self.prefilt_list.append(nn.Sequential(
                nn.Conv1d(in_channels=n_ch[0], out_channels=n_ch[0], kernel_size=15, padding=7, stride=1),
                activation_layer()
            ))
        self.residual = residual

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_ch[0],
                out_channels=n_ch[1],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            activation_layer()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=1, padding=0, stride=1),
            activation_layer()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=1, padding=0, stride=1),
            activation_layer(),
            nn.Dropout(),
            nn.Conv1d(in_channels=n_ch[3], out_channels=n_ch[4], kernel_size=1, padding=0, stride=1)
        )

        self.flatten = nn.Flatten(start_dim=1)

        layers = []
        pre_fc_dim = n_bins_in * n_ch[4]
        for i in range(num_output_layers-1):
            layers.extend([
                ToeplitzLinear(pre_fc_dim, pre_fc_dim),
                activation_layer()
            ])
        self.pre_fc = nn.Sequential(*layers)
        self.fc = ToeplitzLinear(pre_fc_dim, output_dim)

        self.final_norm = nn.Softmax(dim=-1)

        self.register_buffer("abs_shift", torch.zeros((), dtype=torch.long), persistent=True)

    def forward(self, x):
        r"""

        Args:
            x (torch.Tensor): shape (batch, channels, freq_bins)
        """
        x_norm = self.layernorm(x)

        x = self.conv1(x_norm)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_list[p]
            if self.residual:
                x_new = prefilt_layer(x)
                x = x_new + x
            else:
                x = prefilt_layer(x)
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)

        y_pred = self.conv4(conv3_lrelu)
        y_pred = self.flatten(y_pred)
        y_pred = self.pre_fc(y_pred)
        y_pred = self.fc(y_pred)
        return self.final_norm(y_pred)
