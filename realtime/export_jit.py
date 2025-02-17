import torch
import argparse
import datetime
from pesto import load_model


def export_model(checkpoint_name, sampling_rate, chunk_size, script_name):
    """Exports a model using torch.jit.trace and saves it to a file."""
    step_size = 1000 * chunk_size / sampling_rate

    print("Chunk size:", chunk_size)

    model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=4
    )
    model.eval()  # Set the model to evaluation mode

    # Example input for tracing
    example_input = torch.randn(3, chunk_size).clip(-1, 1)

    # Export the model using torch.jit.trace
    traced_model = torch.jit.trace(model, example_input)

    # Save the traced model
    traced_model.save(script_name)
    print(f"Model successfully exported as '{script_name}'")

    return model, script_name


def validate_model(original_model, script_name, chunk_size):
    """Loads the exported model and validates its output."""
    loaded_model = torch.jit.load(script_name)
    loaded_model.eval()

    example_input = torch.randn(2, chunk_size).clip(-1, 1)

    # Run the original and loaded models
    with torch.no_grad():
        original_output = original_model(example_input)
        traced_output = loaded_model(example_input)

    # Compare outputs
    for name, x1, x2 in zip(["pred", "conf", "vol", "act"], original_output, traced_output):
        if torch.allclose(x1, x2):
            print(name, ":", x1.shape, "\n", "Test passed: Outputs are close.\n")
        else:
            print(name, "Test failed: Significant difference in outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model using torch.jit.trace and validate its outputs.")
    parser.add_argument("checkpoint_name", type=str, help="Checkpoint name for loading the model.")
    parser.add_argument("-r", "--sampling_rate", type=int, default=48000, help="Sampling rate of the model.")
    parser.add_argument("-c", "--chunk_size", type=int, default=960, help="Chunk size for processing.")
    parser.add_argument("-s", "--script_name", type=str, default=None, help="Optional custom script name.")

    args = parser.parse_args()

    # Construct default script name if not provided
    if args.script_name is None:
        date_str = datetime.datetime.now().strftime("%y%m%d")
        args.script_name = f"{date_str}_sr{args.sampling_rate//1000}k_h{args.chunk_size}.pt"

    model, script_name = export_model(args.checkpoint_name, args.sampling_rate, args.chunk_size, args.script_name)
    validate_model(model, script_name, args.chunk_size)
