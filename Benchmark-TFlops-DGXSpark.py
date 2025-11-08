#!/usr/bin/env python3

import torch
import time
import sys

# --- Configuration ---
# Matrix size (M, K, N). Powers of 2 are best for Tensor Cores.
# 8192 is a large test. Use 4096 if you get memory errors.
MATRIX_SIZE = 8192
N_WARMUP = 10
N_ITERATIONS = 1000

# Precision settings to test
PRECISION_NAME = "FP16"
PRECISION_DTYPE = torch.float16
# ---------------------

def test_tflops():
    print(f"PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"Running on device: {torch.cuda.get_device_name(device)}")
    print(f"Testing {PRECISION_NAME} performance with {N_ITERATIONS} iterations...\n")

    # Define matrix dimensions
    M, K, N = MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE

    # Create random matrices on the GPU with the specified precision
    try:
        a = torch.randn(M, K, device=device, dtype=PRECISION_DTYPE)
        b = torch.randn(K, N, device=device, dtype=PRECISION_DTYPE)
    except torch.cuda.OutOfMemoryError:
        print(f"Error: Out of memory. Try a smaller MATRIX_SIZE (e.g., 4096) at the top of the script.")
        return
    except Exception as e:
        print(f"An error occurred creating tensors: {e}")
        return

    # --- Warm-up Run ---
    # This compiles the CUDA kernel so we don't time the compilation.
    print("Performing warm-up runs...")
    for _ in range(N_WARMUP):
        _ = torch.matmul(a, b)
    
    # Ensure all kernels are finished before starting the timer
    torch.cuda.synchronize()

    # --- Timed Benchmark ---
    print(f"Running timed benchmark ({N_ITERATIONS} iterations)...")
    start_time = time.time()
    for _ in range(N_ITERATIONS):
        _ = torch.matmul(a, b)
    
    # Wait for all operations to complete
    # --- THIS IS THE CORRECTED LINE ---
    torch.cuda.synchronize()
    end_time = time.time()
    
    # --- Calculations ---
    total_time = end_time - start_time
    avg_time_per_iter = total_time / N_ITERATIONS
    
    # A matrix multiplication (GEMM) has 2 * M * K * N operations
    # (Multiply-Add is 2 operations)
    operations = 2 * M * K * N
    
    # FLOPS = operations per second
    flops = operations / avg_time_per_iter
    
    # TFLOPS = FLOPS / 1 trillion
    tflops = flops / 1_000_000_000_000

    print("\n--- Results ---")
    print(f"  Precision Tested:  {PRECISION_NAME} ({PRECISION_DTYPE})")
    print(f"  Matrix Dimensions: {M}x{K} @ {K}x{N}")
    print(f"  Average Time:      {avg_time_per_iter * 1000:.2f} ms")
    print(f"  Achieved Value:    {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    test_tflops()