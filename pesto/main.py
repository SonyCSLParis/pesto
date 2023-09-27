from pesto.parser import parse_args
from pesto.predict import predict_from_files


def pesto():
    args = parse_args()
    predict_from_files(**vars(args))


if __name__ == "__main__":
    pesto()
