import argparse
import time
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark an ONNX model.")
    parser.add_argument("onnx_model", type=str, help="Path to the ONNX model file.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1, help="Batch size for inference."
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=256, help="Size of audio chunks."
    )
    return parser.parse_args()


def benchmark_model(onnx_model, batch_size, chunk_size, num_iterations=5000):
    # Load the exported ONNX model
    session = ort.InferenceSession(onnx_model)

    example_audio = (
        np.random.randn(batch_size, chunk_size).clip(-1, 1).astype(np.float32)
    )

    # Initialize cache state for the ONNX model
    cache_state = np.zeros(
        (batch_size, session.get_inputs()[1].shape[1]), dtype=np.float32
    )

    # Measure inference time
    t_list = []
    for _ in range(num_iterations):
        outputs = session.run(
            None, {"audio": example_audio, "cache": cache_state}
        )
        # Update cache state for next iteration
        cache_state = outputs[4]  # cache_out is the 5th output (index 4)
        t_list.append(time.time())

    t_list = np.array(t_list, dtype=np.float64)
    it = t_list[1:] - t_list[:-1]

    print("Compute time per chunk:")
    print(f"\t{1000 * it.mean():.3f} ± {1000 * it.std():.3f} ms")

    it = 1 / it
    print("FPS:")
    print(f"\t{it.mean():.3f} ± {it.std():.3f} FPS")


if __name__ == "__main__":
    args = parse_args()
    benchmark_model(args.onnx_model, args.batch_size, args.chunk_size)
