import os
import argparse
import numpy as np
import torch
import onnxruntime as ort
import torchaudio
import matplotlib.pyplot as plt
from pesto import load_model

# Default checkpoint name
CHECKPOINT = "mir-1k_g7"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test accuracy across PyTorch, TorchScript, and ONNX models"
    )
    parser.add_argument(
        "-r",
        "--sampling_rate",
        type=int,
        default=44100,
        help="Sampling rate (default: 44100)",
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=1024, help="Chunk size (default: 1024)"
    )
    return parser.parse_args()


def load_audio(target_sr):
    """Load and preprocess test audio."""
    audio, sr = torchaudio.load("tests/audios/example.wav")
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    return audio[0].numpy().astype(np.float32), target_sr


def process_chunks(audio, chunk_size, hop_size=None):
    """Split audio into chunks."""
    if hop_size is None:
        hop_size = chunk_size
    chunks = []
    for i in range(0, len(audio) - chunk_size + 1, hop_size):
        chunks.append(audio[i : i + chunk_size])
    return np.array(chunks)


def run_model(model_type, chunks, sr, chunk_size):
    """Run model inference on chunks."""
    results = []

    if model_type == "pytorch":
        step_size = chunk_size / sr * 1000
        model = load_model(
            CHECKPOINT,
            step_size=step_size,
            sampling_rate=sr,
            streaming=True,
            mirror=1.0,
        )
        model.eval()

        with torch.no_grad():
            for chunk in chunks:
                pred, conf, vol, _ = model(torch.from_numpy(chunk).unsqueeze(0))
                results.append([pred[0, 0].item(), conf[0, 0].item(), vol[0, 0].item()])

    elif model_type == "torchscript":
        model_path = f"{CHECKPOINT}_{sr}_{chunk_size}.pt"
        if not os.path.exists(model_path):
            print(f"TorchScript model not found: {model_path}")
            return None

        model = torch.jit.load(model_path)
        model.eval()

        with torch.no_grad():
            for chunk in chunks:
                pred, conf, vol, _ = model(torch.from_numpy(chunk).unsqueeze(0))
                results.append([pred[0, 0].item(), conf[0, 0].item(), vol[0, 0].item()])

    elif model_type == "onnx":
        model_path = f"{CHECKPOINT}_{sr}_{chunk_size}.onnx"
        if not os.path.exists(model_path):
            print(f"ONNX model not found: {model_path}")
            return None

        session = ort.InferenceSession(model_path)
        cache_state = np.zeros((1, session.get_inputs()[1].shape[1]), dtype=np.float32)

        for chunk in chunks:
            outputs = session.run(
                None,
                {"audio": chunk.reshape(1, -1), "cache": cache_state},
            )
            results.append([outputs[0][0, 0], outputs[1][0, 0], outputs[2][0, 0]])
            cache_state = outputs[4]  # cache_out

    return np.array(results)


def main():
    args = parse_args()

    print(f"Loading audio (target SR: {args.sampling_rate}Hz)...")
    audio, sr = load_audio(args.sampling_rate)
    chunks = process_chunks(audio, args.chunk_size)
    print(f"Processing {len(chunks)} chunks of size {args.chunk_size}...")

    # Run all available models
    model_results = {}

    # PyTorch (always available)
    print("Running PyTorch model...")
    pytorch_results = run_model("pytorch", chunks, sr, args.chunk_size)
    model_results["PyTorch"] = pytorch_results

    # TorchScript (check if exists)
    print("Running TorchScript model...")
    torchscript_results = run_model("torchscript", chunks, sr, args.chunk_size)
    if torchscript_results is not None:
        model_results["TorchScript"] = torchscript_results

    # ONNX (check if exists)
    print("Running ONNX model...")
    onnx_results = run_model("onnx", chunks, sr, args.chunk_size)
    if onnx_results is not None:
        model_results["ONNX"] = onnx_results

    if len(model_results) < 2:
        print("Need at least 2 models to compare. Exiting.")
        return

    # Convert to Hz and calculate differences
    def midi_to_hz(midi):
        return 440 * (2 ** ((midi - 69) / 12))

    model_hz = {}
    for name, results in model_results.items():
        model_hz[name] = midi_to_hz(results[:, 0])

    # Create clean plot
    time_axis = np.arange(len(chunks)) * args.chunk_size / sr

    plt.figure(figsize=(12, 6))

    # Pitch predictions
    plt.subplot(2, 1, 1)
    colors = ["b-", "r-", "g--", "m:", "c-"]
    for i, (name, results) in enumerate(model_results.items()):
        plt.plot(time_axis, results[:, 0], colors[i], label=name, alpha=0.8)
    plt.ylabel("MIDI Note")
    plt.title("Pitch Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confidence values
    plt.subplot(2, 1, 2)
    for i, (name, results) in enumerate(model_results.items()):
        plt.plot(time_axis, results[:, 1], colors[i], label=name, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.title("Confidence Values")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    plt.show()

    # Print accuracy summary
    print("\nAccuracy Summary:")
    model_names = list(model_results.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            hz1, hz2 = model_hz[name1], model_hz[name2]
            avg_diff = np.mean(np.abs(hz1 - hz2))
            print(f"{name1} vs {name2}: {avg_diff:.3f} Hz avg")


if __name__ == "__main__":
    main()
