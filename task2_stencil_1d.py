import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:

        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]

        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]

        center = d_in[idx]

        d_out[idx] = (
            0.25 * left
            + 0.5 * center
            + 0.25 * right
        )

def run_stencil(h_in):
    N = len(h_in)

    if N == 0:
        return np.empty(0, dtype=np.float32)

    h_in = np.asarray(h_in, dtype=np.float32)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array(N, dtype=np.float32)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    stencil_1d[blocks_per_grid, threads_per_block](
        d_in, d_out, N
    )

    cuda.synchronize()

    return d_out.copy_to_host()

def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')

    return (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )

def main():
    N = 100007

    h_in = np.random.rand(N).astype(np.float32)

    h_out_gpu = run_stencil(h_in)

    cpu_ref = cpu_stencil(h_in)

    assert np.allclose(
        h_out_gpu,
        cpu_ref,
        atol=1e-4
    ), "GPU and CPU results do not match!"

    delta = np.max(np.abs(h_out_gpu - cpu_ref))

    print("TASK 2: 1D STENCIL")
    print("-------------------------")
    print("Array size:", N)
    print(f"TASK 2 PASSED: MAX DELTA = {delta:.8f}")


if __name__ == "__main__":
    main()