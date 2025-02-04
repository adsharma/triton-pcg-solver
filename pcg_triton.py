import triton
import triton.language as tl


# Define the PCG kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["num_rows"],
)
@triton.jit
def pcg_kernel(
    A_values,
    A_row_offsets,
    A_column_indices,
    b,
    x,
    r,
    p,
    z,
    tmp,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the global thread ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE

    # Iterate over the rows of the matrix
    for row in range(block_start, block_end):
        if row < num_rows:
            # Initialize the temporary variable
            tmp[row] = 0.0

            # Perform the SpMV operation
            row_start = A_row_offsets[row]
            row_end = A_row_offsets[row + 1]
            for i in range(row_start, row_end):
                tmp[row] += A_values[i] * x[A_column_indices[i]]

            # Update the residual vector
            r[row] = b[row] - tmp[row]

            # Update the preconditioned residual vector
            z[row] = r[row] / tmp[row]

            # Update the direction vector
            p[row] = z[row] + (r[row] * r[row]) / (p[row] * p[row]) * p[row]


# Define the host function to launch the PCG kernel
def launch_pcg_kernel(
    A_values, A_row_offsets, A_column_indices, b, x, r, p, z, tmp, num_rows
):
    grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_SIZE"]),)
    pcg_kernel[grid](
        A_values, A_row_offsets, A_column_indices, b, x, r, p, z, tmp, num_rows
    )
