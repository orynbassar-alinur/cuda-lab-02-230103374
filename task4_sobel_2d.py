import numpy as np
from numba import cuda

@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):

    col, row = cuda.grid(2)

    if row < rows and col < cols:

        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            d_out[row, col] = 0.0

        else:
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1]
                + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1]
                + 1.0 * d_in[row + 1, col + 1]
            )

def run_sobel(h_img):

    h_img = np.asarray(h_img, dtype=np.float32)

    if h_img.ndim != 2:
        raise ValueError("Input image must be a 2D matrix")

    rows, cols = h_img.shape

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)

    threads_2d = (16, 16)

    blocks_x = (cols + threads_2d[0] - 1) // threads_2d[0]
    blocks_y = (rows + threads_2d[1] - 1) // threads_2d[1]

    blocks_2d = (blocks_x, blocks_y)

    sobel_x_kernel[blocks_2d, threads_2d](
        d_in,
        d_out,
        rows,
        cols
    )

    cuda.synchronize()

    return d_out.copy_to_host()

def cpu_sobel_x(img):

    img = np.asarray(img, dtype=np.float32)
    rows, cols = img.shape

    out = np.zeros((rows, cols), dtype=np.float32)

    out[1:-1, 1:-1] = (
        -1.0 * img[:-2, :-2]
        + 1.0 * img[:-2, 2:]
        - 2.0 * img[1:-1, :-2]
        + 2.0 * img[1:-1, 2:]
        - 1.0 * img[2:, :-2]
        + 1.0 * img[2:, 2:]
    )

    return out


def main():

    rows = 2048
    cols = 2048

    print("TASK 4: 2D SOBEL-X FILTER")
    print("--------------------------------")

    h_img = np.random.rand(rows, cols).astype(np.float32)

    print("Input image shape:", h_img.shape)
    print("Threads per block:", (16, 16))

    h_out_gpu = run_sobel(h_img)

    h_out_cpu = cpu_sobel_x(h_img)

    assert np.allclose(
        h_out_gpu,
        h_out_cpu,
        atol=1e-4
    ), "Task 4 FAILED: GPU output does not match CPU reference!"

    assert np.all(h_out_gpu[0, :] == 0.0), "Top border is not zero"
    assert np.all(h_out_gpu[-1, :] == 0.0), "Bottom border is not zero"
    assert np.all(h_out_gpu[:, 0] == 0.0), "Left border is not zero"
    assert np.all(h_out_gpu[:, -1] == 0.0), "Right border is not zero"

    delta = np.max(np.abs(h_out_gpu - h_out_cpu))

    print("Output image shape:", h_out_gpu.shape)
    print(f"TASK 4 PASSED: MAX DELTA = {delta:.8f}")
    print("--------------------------------")


if __name__ == "__main__":
    main()