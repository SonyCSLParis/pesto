import torch

from pesto import load_model

# options
STEP_SIZE = 20.
SAMPLING_RATE = 48000
MIRROR = 1.

CHUNK_SIZE = int(STEP_SIZE * SAMPLING_RATE / 1000 + 0.5)

CHECKPOINT_NAME = "mir-1k_g5_conf"
SCRIPT_NAME = "1009.pt"

model = load_model(CHECKPOINT_NAME,
                   step_size=STEP_SIZE,
                   sampling_rate=SAMPLING_RATE,
                   streaming=True,
                   mirror=MIRROR)
model.eval()  # Set the model to evaluation mode

# Example input for tracing (shape should match what your model expects)
example_input = torch.randn(CHUNK_SIZE).clip(-1, 1)  # Modify according to your input shape

# Export the model using torch.jit.trace
traced_model = torch.jit.trace(model, example_input)

# Save the traced model to a file
traced_model.save(SCRIPT_NAME)
print(f"Model successfully exported as '{SCRIPT_NAME}'")

# Load the exported TorchScript model
loaded_model = torch.jit.load(SCRIPT_NAME)
loaded_model.eval()  # Make sure it's in evaluation mode

example_input = torch.randn(CHUNK_SIZE).clip(-1, 1)  # Modify according to your input shape

# Run the original model and the loaded model
with torch.no_grad():
    original_output = model(example_input)
    traced_output = loaded_model(example_input)

# Test if the outputs are close
for name, x1, x2 in zip(["pred", "conf", "vol", "act"], original_output, traced_output):
    if torch.allclose(x1, x2):
        print(name, "Test passed: The traced model outputs are close to the original model outputs.")
    else:
        print(name, "Test failed: There is a significant difference between the traced and original model outputs.")
