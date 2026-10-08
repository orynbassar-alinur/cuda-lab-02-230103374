import numpy as np
from numba import cuda

@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):

    start = cuda.grid(1)

    stride = cuda.gridsize(1)

    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor

def run_grid_stride(h_arr, factor):

    N = len(h_arr)

    if N == 0:
        return np.empty(0, dtype=np.float32)

    h_arr = np.asarray(h_arr, dtype=np.float32)

    d_arr = cuda.to_device(h_arr)

    threads_per_block = 256
    blocks_per_grid = 64

    grid_stride_scale_kernel[
        blocks_per_grid,
        threads_per_block
    ](d_arr, factor, N)

    cuda.synchronize()

    return d_arr.copy_to_host()

def main():

    N = 2**24
    factor = 4.25

    h_arr = np.ones(N, dtype=np.float32)

    print("TASK 3: GRID-STRIDE SCALING")
    print("--------------------------------")

    print("Array size:", N)
    print("Threads per block:", 256)
    print("Blocks per grid:", 64)
    print("Total CUDA threads:", 256 * 64)

    result = run_grid_stride(h_arr, factor)

    assert np.allclose(
        result,
        factor
    ), "Task 3 FAILED: Incorrect scaling!"

    print("--------------------------------")
    print("TASK 3 PASSED!")
    print("All", N, "elements equal", factor)


if __name__ == "__main__":
    main()