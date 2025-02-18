import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog='PESTO', description='Efficient pitch estimation')
    parser.add_argument('audio_files', metavar='FILE', type=str, nargs='+',
                        help='audio files to process')
    parser.add_argument('-m', '--model_name', type=str, default="mir-1k_g7", choices=["mir-1k_g7"],
                        help='choice of the model')
    parser.add_argument('-o', '--output', metavar='DIR', type=str, default=None,
                        help='directory to save the output predictions')
    parser.add_argument('-e', '--export_format', metavar='FMT', type=str, default=["csv"], choices=["csv", "npz", "png"], nargs='+')
    parser.add_argument('-r', '--reduction', type=str, default='alwa', choices=["alwa", "argmax", "mean"],
                        help='how to predict pitch from output probability distributions')
    parser.add_argument('-s', '--step_size', type=float, default=10.,
                        help='the step size in milliseconds between each prediction')
    parser.add_argument('-F', '--no_convert_to_freq', action='store_true',
                        help='if true, does not convert the predicted pitch to frequency domain and '
                             'returns predictions as semitones')
    parser.add_argument('-c', '--num_chunks', type=int, default=1,
                        help='number of chunks to split the input data into (default: 1). '
                             'Can be increased to prevent out-of-memory errors.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='the index of the GPU to use, -1 for CPU')
    return parser.parse_args()
