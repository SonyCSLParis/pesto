from functools import partial
from math import log
from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import CropCQT
from .utils import reduce_activations


OUTPUT_TYPE = Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


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


class Resnet1d(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report,
    but for longer HCQT input (always stride 1 in time)
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).
    Outputs one channel with sigmoid activation.

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:     Number of input channels (harmonics in HCQT)
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        residual:         If True, use residual connections for prefiltering (default: False)
        n_bins_in:        Number of input bins (12 * number of octaves)
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch, num_octaves * 12)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """

    def __init__(self,
                 n_chan_input=1,
                 n_chan_layers=(20, 20, 10, 1),
                 n_prefilt_layers=1,
                 prefilt_kernel_size=15,
                 residual=False,
                 n_bins_in=216,
                 output_dim=128,
                 activation_fn: str = "leaky",
                 a_lrelu=0.3,
                 p_dropout=0.2,
                 **unused):
        super(Resnet1d, self).__init__()

        self.hparams = dict(n_chan_input=n_chan_input,
                            n_chan_layers=n_chan_layers,
                            n_prefilt_layers=n_prefilt_layers,
                            prefilt_kernel_size=prefilt_kernel_size,
                            residual=residual,
                            n_bins_in=n_bins_in,
                            output_dim=output_dim,
                            activation_fn=activation_fn,
                            a_lrelu=a_lrelu,
                            p_dropout=p_dropout)

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        n_in = n_chan_input
        n_ch = n_chan_layers
        if len(n_ch) < 5:
            n_ch.append(1)

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_in,
                      out_channels=n_ch[0],
                      kernel_size=prefilt_kernel_size,
                      padding=prefilt_padding,
                      stride=1),
            activation_layer(),
            nn.Dropout(p=p_dropout)
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=n_ch[0],
                          out_channels=n_ch[0],
                          kernel_size=prefilt_kernel_size,
                          padding=prefilt_padding,
                          stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            )
            for _ in range(n_prefilt_layers-1)
        ])
        self.residual = residual

        conv_layers = []
        for i in range(len(n_chan_layers)-1):
            conv_layers.extend([
                nn.Conv1d(in_channels=n_ch[i],
                          out_channels=n_ch[i + 1],
                          kernel_size=1,
                          padding=0,
                          stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)

        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""

        Args:
            x (torch.Tensor): shape (batch, channels, freq_bins)
        """

        # compute pitch predictions
        x = self.layernorm(x)

        x = self.conv1(x)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]
            if self.residual:
                x_new = prefilt_layer(x)
                x = x_new + x
            else:
                x = prefilt_layer(x)

        x = self.conv_layers(x)
        x = self.flatten(x)

        y_pred = self.fc(x)

        return self.final_norm(y_pred)


class ConfidenceClassifier(nn.Module):
    r"""A simple pre-trained classifier that returns whether a sample is voiced or not

    # TODO: add args for this module, it should not be hardcoded
    """
    def __init__(self):
        super(ConfidenceClassifier, self).__init__()
        self.conv = nn.Conv1d(1, 1, 39, stride=3)
        self.linear = nn.Linear(72, 1)

    def forward(self, x):
        r"""Computes the confidence in [0, 1] that a frame is voiced or not

        Args:
            x (torch.Tensor): shape (batch_size, channels, freq_bins)

        Returns:
            torch.Tensor: confidence in [0, 1], shape (batch_size,)
        """
        geometric_mean = x.log().mean(dim=-1, keepdim=True).exp()
        artithmetric_mean = x.mean(dim=-1, keepdim=True).clip_(min=1e-8)
        flatness = geometric_mean / artithmetric_mean

        x = F.relu(self.conv(x.unsqueeze(1)).squeeze(1))
        return torch.sigmoid(self.linear(torch.cat((x, flatness), dim=-1))).squeeze(-1)


class PESTO(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 preprocessor: nn.Module,
                 crop_kwargs: Optional[Mapping[str, Any]] = None,
                 reduction: str = "alwa"):
        super(PESTO, self).__init__()
        self.encoder = encoder
        self.preprocessor = preprocessor

        # TODO: make this clean
        self.confidence = ConfidenceClassifier()

        # crop CQT
        if crop_kwargs is None:
            crop_kwargs = {}
        self.crop_cqt = CropCQT(**crop_kwargs)

        self.reduction = reduction

        # constant shift to get absolute pitch from predictions
        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

    def forward(self,
                audio_waveforms: torch.Tensor,
                sr: Optional[int] = None,
                convert_to_freq: bool = False,
                return_activations: bool = True) -> OUTPUT_TYPE:
        r"""

        Args:
            audio_waveforms (torch.Tensor): mono audio waveform or batch of mono audio waveforms,
                shape (batch_size?, num_samples)
            sr (int, optional): sampling rate, defaults to the previously used sampling rate
            convert_to_freq (bool): whether to convert the result to frequencies or return fractional semitones instead.
            return_activations (bool): whether to return activations or pitch predictions only

        Returns:
            preds (torch.Tensor): pitch predictions in SEMITONES, shape (batch_size?, num_timesteps)
                where `num_timesteps` ~= `num_samples` / (`self.hop_size` * `sr`)
            confidence (torch.Tensor): confidence of whether frame is voiced or unvoiced in [0, 1],
                shape (batch_size?, num_timesteps)
            activations (torch.Tensor): activations of the model, shape (batch_size?, num_timesteps, output_dim)
        """
        batch_size = audio_waveforms.size(0) if audio_waveforms.ndim == 2 else None
        x = self.preprocessor(audio_waveforms, sr=sr).flatten(0, 1)

        # compute volume and confidence
        energy = x.mul_(log(10) / 10.).exp().squeeze_(1)
        vol = energy.sum(dim=-1)  # .log10_().mul_(20)

        confidence = self.confidence(energy)

        x = self.crop_cqt(x)  # the CQT has to be cropped beforehand

        activations = self.encoder(x)

        if batch_size is None:
            confidence.squeeze_(0)
        else:
            activations = activations.view(batch_size, -1, activations.size(-1))
            confidence = confidence.view(batch_size, -1)
            vol = vol.view(batch_size, -1)

        activations = activations.roll(-round(self.shift.cpu().item() * self.bins_per_semitone), -1)

        preds = reduce_activations(activations, reduction=self.reduction)

        if convert_to_freq:
            preds = 440 * 2 ** ((preds - 69) / 12)

        if return_activations:
            return preds, confidence, vol, activations

        return preds, confidence, vol

    @property
    def bins_per_semitone(self) -> int:
        return self.preprocessor.hcqt_kwargs["bins_per_semitone"]

    @property
    def hop_size(self) -> float:
        r"""Returns the hop size of the model (in milliseconds)"""
        return self.preprocessor.hop_size
