from parser import parse_args
from predict import predict_from_files


if __name__ == "__main__":
    args = parse_args()
    predict_from_files(**vars(args))
