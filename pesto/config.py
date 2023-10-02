# params
bins_per_semitone = 3
fmin = 27.5  # A0
center_bins = True


if center_bins:
    fmin = fmin / (2 ** ((bins_per_semitone - 1) / (12 * bins_per_semitone)))


model_args = dict(
    n_chan_layers=(40, 30, 30, 10, 3),
    n_prefilt_layers=2,
    residual=True,
    n_bins_in=88*bins_per_semitone,
    output_dim=128*bins_per_semitone
)


cqt_args = dict(
    bins_per_semitone=bins_per_semitone,
    fmin=fmin,
    n_bins=99*bins_per_semitone,  # maximal number of semitones when working at 16 kHz
    output_format="Complex",
    center=True
)
