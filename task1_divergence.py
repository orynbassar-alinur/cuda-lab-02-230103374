import numpy as np
import time
from numba import cuda

N = 2**20
THREADS_PER_BLOCK = 256
ITERATIONS = 1000

@cuda.jit
def kernel_a(d_arr):
    idx = cuda.grid(1)

    if idx < d_arr.size:
        y = d_arr[idx]

        for i in range(ITERATIONS):
            y = y * 1.0001 + 0.0001

        d_arr[idx] = y

@cuda.jit
def kernel_b(d_arr):
    idx = cuda.grid(1)

    if idx < d_arr.size:
        y = d_arr[idx]

        if idx % 2 == 0:
            for i in range(ITERATIONS):
                y = y * 1.0001 + 0.0001
        else:
            for i in range(ITERATIONS):
                y = (y - 0.0001) / 1.0001

        d_arr[idx] = y

@cuda.jit
def kernel_c(d_arr):
    idx = cuda.grid(1)

    if idx < d_arr.size:
        y = d_arr[idx]

        warp_id = idx // 32

        if warp_id % 2 == 0:
            for i in range(ITERATIONS):
                y = y * 1.0001 + 0.0001
        else:
            for i in range(ITERATIONS):
                y = (y - 0.0001) / 1.0001

        d_arr[idx] = y

def benchmark(kernel, d_arr):
    blocks_per_grid = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    kernel[blocks_per_grid, THREADS_PER_BLOCK](d_arr)
    cuda.synchronize()

    times = []

    for _ in range(10):
        start = time.perf_counter()

        kernel[blocks_per_grid, THREADS_PER_BLOCK](d_arr)
        cuda.synchronize()

        end = time.perf_counter()

        times.append((end - start) * 1000)

    return np.mean(times)


def main():
    h_arr = np.ones(N, dtype=np.float32)
    d_arr = cuda.to_device(h_arr)

    print("TASK 1: WARP DIVERGENCE BENCHMARK")
    print("----------------------------------")

    time_a = benchmark(kernel_a, d_arr)
    time_b = benchmark(kernel_b, d_arr)
    time_c = benchmark(kernel_c, d_arr)

    print(f"Kernel A (Uniform Path): {time_a:.4f} ms")
    print(f"Kernel B (Full Divergence): {time_b:.4f} ms")
    print(f"Kernel C (Warp-Aligned): {time_c:.4f} ms")

    print("----------------------------------")
    print("TASK 1 COMPLETED")


if __name__ == "__main__":
    main()