import argparse
import time
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a TorchScript model.")
    parser.add_argument("script_name", type=str, help="Path to the TorchScript model file.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("-c", "--chunk_size", type=int, default=256, help="Size of audio chunks.")
    return parser.parse_args()


def benchmark_model(script_name, batch_size, chunk_size, num_iterations=5000):
    # Load the exported TorchScript model
    model = torch.jit.load(script_name)
    model.eval()  # Ensure it's in evaluation mode

    example_input = torch.randn(batch_size, chunk_size).clip(-1, 1)

    # Measure inference time
    t_list = []
    for _ in range(num_iterations):
        _ = model(example_input)
        t_list.append(time.time())

    t_list = torch.tensor(t_list, dtype=torch.float64)
    it = t_list[1:] - t_list[:-1]

    print("Compute time per chunk:")
    print(f"\t{1000 * it.mean().item():.3f} ± {1000 * it.std().item():.3f} ms")

    it = 1 / it
    print("FPS:")
    print(f"\t{it.mean().item():.3f} ± {it.std().item():.3f} FPS")


if __name__ == "__main__":
    args = parse_args()
    benchmark_model(args.script_name, args.batch_size, args.chunk_size)
