import torch
import argparse
import numpy as np
import onnx
import onnxruntime as ort

from pesto import load_model
from realtime.onnx_wrapper import StatelessPESTO


def export_model(checkpoint_name, sampling_rate, chunk_size, script_name, batch_size):
    """Exports a model using ONNX with proper cache state management."""
    step_size = 1000 * chunk_size / sampling_rate

    print("Chunk size:", chunk_size)

    # Load the original model
    model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=batch_size,
        mirror=1.0,
    )
    model.eval()

    # Wrap it with the stateless wrapper
    stateless_model = StatelessPESTO(model)
    stateless_model.eval()

    # Example inputs for tracing
    example_audio = torch.randn(batch_size, chunk_size).clip(-1, 1)
    example_cache = stateless_model.init_cache(batch_size=batch_size, device="cpu")

    print(f"Cache size: {example_cache.numel()} elements")
    print(f"Audio input shape: {example_audio.shape}")

    # Export the model using torch.onnx.export
    torch.onnx.export(
        stateless_model,
        (example_audio, example_cache),
        script_name,
        input_names=["audio", "cache"],
        output_names=[
            "prediction",
            "confidence",
            "volume",
            "activations",
            "cache_out",
        ],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "audio_length"},
            "prediction": {0: "batch_size", 1: "time_steps"},
            "confidence": {0: "batch_size", 1: "time_steps"},
            "volume": {0: "batch_size", 1: "time_steps"},
            "activations": {0: "batch_size", 1: "time_steps"},
        },
    )

    print(f"Model successfully exported as '{script_name}'")
    return stateless_model, script_name


def validate_model(checkpoint_name, sampling_rate, script_name, chunk_size, batch_size):
    """Loads the exported model and validates its output using a torch model."""
    loaded_model = onnx.load(script_name)
    onnx.checker.check_model(loaded_model)

    print("ONNX model validation successful!")
    print(
        f"Model has {len(loaded_model.graph.input)} input(s) and {len(loaded_model.graph.output)} output(s)"
    )

    # Print input/output info
    for i, input_info in enumerate(loaded_model.graph.input):
        print(f"Input {i}: {input_info.name}")

    for i, output_info in enumerate(loaded_model.graph.output):
        print(f"Output {i}: {output_info.name}")

    # Create ONNX Runtime session
    session = ort.InferenceSession(script_name)

    # Load a model for comparison
    step_size = 1000 * chunk_size / sampling_rate
    torch_model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=batch_size,
        mirror=1.0,
    )
    torch_model.eval()

    example_input = torch.randn(batch_size, chunk_size).clip(-1, 1)

    # Run the normal PyTorch model (no external cache management)
    with torch.no_grad():
        pytorch_output = torch_model(example_input)

    # Create dummy cache state for ONNX (zeros)
    cache_size = session.get_inputs()[1].shape[1]
    cache_state = torch.zeros((batch_size, cache_size), dtype=torch.float32)

    # Convert to numpy for ONNX
    onnx_output = session.run(
        None,
        {"audio": example_input.numpy(), "cache": cache_state.numpy()},
    )

    # Compare outputs
    output_names = ["pred", "conf", "vol", "act"]
    onnx_comparison_outputs = onnx_output[:4]  # Skip cache_out for comparison
    for name, torch_out, onnx_out in zip(
        output_names, pytorch_output, onnx_comparison_outputs
    ):
        torch_out_np = torch_out.cpu().numpy()
        if np.allclose(torch_out_np, onnx_out, rtol=1e-4, atol=1e-5):
            print(name, ":", torch_out.shape, "\n", "Test passed: Outputs are close.\n")
        else:
            max_diff = np.max(np.abs(torch_out_np - onnx_out))
            print(f"{name}: Test failed: Max diff = {max_diff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a model using ONNX and validate its outputs."
    )
    parser.add_argument(
        "checkpoint_name", type=str, help="Checkpoint name for loading the model."
    )
    parser.add_argument(
        "-r",
        "--sampling_rate",
        type=int,
        default=48000,
        help="Sampling rate of the model.",
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=960, help="Chunk size for processing."
    )
    parser.add_argument(
        "-s",
        "--script_name",
        type=str,
        default=None,
        help="Optional custom script name.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2,
        help="Max batch size for the ONNX model (default: 2).",
    )

    args = parser.parse_args()

    # Construct default script name if not provided
    if args.script_name is None:
        args.script_name = (
            f"{args.checkpoint_name}_{args.sampling_rate}_{args.chunk_size}.onnx"
        )

    model, script_name = export_model(
        args.checkpoint_name,
        args.sampling_rate,
        args.chunk_size,
        args.script_name,
        args.batch_size,
    )
    validate_model(
        args.checkpoint_name,
        args.sampling_rate,
        script_name,
        args.chunk_size,
        args.batch_size,
    )
