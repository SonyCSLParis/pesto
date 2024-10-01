import torch
import torch.nn as nn
import time


# Function to benchmark 2D convolution (with kernel size (k, 2))
def benchmark_2d_conv(input_data, kernel_size, num_channels, device):
    conv2d = nn.Conv2d(1, num_channels, kernel_size=(kernel_size, 2), bias=False).to(device)

    # Reshape the input to (batch_size, 1, seq_len, 1) to simulate 2D data
    input_data_reshaped = input_data.unsqueeze(-1).expand(*input_data.size(), 2).to(device)

    # Warm-up run
    with torch.no_grad():
        output = conv2d(input_data_reshaped)

    # Measure time on GPU or CPU
    if device == torch.device('cuda'):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        # Run the convolution
        with torch.no_grad():
            output = conv2d(input_data_reshaped)

        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)  # Returns time in milliseconds
    else:
        start_time = time.time()
        with torch.no_grad():
            output = conv2d(input_data_reshaped)
        end_time = time.time()
        return (end_time - start_time) * 1000  # Return time in milliseconds


# Function to benchmark 1D convolution (with doubled channels)
def benchmark_1d_conv(input_data, kernel_size, num_channels, device):
    conv1d = nn.Conv1d(1, 2 * num_channels, kernel_size=kernel_size, bias=False).to(device)

    # Move input data to the appropriate device
    input_data = input_data.to(device)

    # Warm-up run
    with torch.no_grad():
        output = conv1d(input_data)

    # Measure time on GPU or CPU
    if device == torch.device('cuda'):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        # Run the convolution
        with torch.no_grad():
            output = conv1d(input_data)

        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)  # Returns time in milliseconds
    else:
        start_time = time.time()
        with torch.no_grad():
            output = conv1d(input_data)
        end_time = time.time()
        return (end_time - start_time) * 1000  # Return time in milliseconds


# Main benchmarking function
def main(k=8192, batch_size=32, seq_length=32768, num_channels=16, num_runs=10):
    # Create random input data (Batch size x Channels x Seq_length) for 1D conv
    input_1d = torch.randn(batch_size, 1, seq_length)

    # Benchmark on CPU
    print("Benchmarking on CPU...")
    time_2d_cpu = sum(
        benchmark_2d_conv(input_1d, k, num_channels, device=torch.device('cpu')) for _ in range(num_runs)) / num_runs
    time_1d_cpu = sum(
        benchmark_1d_conv(input_1d, k, num_channels, device=torch.device('cpu')) for _ in range(num_runs)) / num_runs
    print(f"Avg time for 2D convolution on CPU: {time_2d_cpu:.3f} ms")
    print(f"Avg time for 1D convolution on CPU: {time_1d_cpu:.3f} ms")

    # Check if GPU is available
    if torch.cuda.is_available():
        print("\nBenchmarking on GPU...")
        input_1d = input_1d.to(torch.device('cuda'))

        time_2d_gpu = sum(benchmark_2d_conv(input_1d, k, num_channels, device=torch.device('cuda')) for _ in
                          range(num_runs)) / num_runs
        time_1d_gpu = sum(benchmark_1d_conv(input_1d, k, num_channels, device=torch.device('cuda')) for _ in
                          range(num_runs)) / num_runs
        print(f"Avg time for 2D convolution on GPU: {time_2d_gpu:.3f} ms")
        print(f"Avg time for 1D convolution on GPU: {time_1d_gpu:.3f} ms")
    else:
        print("\nGPU not available.")


if __name__ == "__main__":
    main()