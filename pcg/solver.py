import torch
import triton
import triton.language as tl
import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional


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
    max_iterations,
    tolerance,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Preconditioned Conjugate Gradient kernel for solving linear systems Ax = b
    Uses diagonal preconditioning and minimizes control flow divergence
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, num_rows)

    # Initialize residual
    for row in range(block_start, block_end):
        r_val = tl.load(b + row)
        start = tl.load(A_row_offsets + row)
        end = tl.load(A_row_offsets + row + 1)
        for j in range(start, end):
            col = tl.load(A_column_indices + j)
            a_val = tl.load(A_values + j)
            x_val = tl.load(x + col)
            r_val -= a_val * x_val
        tl.store(r + row, r_val)

    tl.debug_barrier()

    # Iteration variables
    rho = 0.0
    rho_prev = 0.0
    converged = False

    # Main PCG iteration loop
    for iteration in range(max_iterations):
        active = not converged

        # Reset reduction variables using masked store
        tl.store(tmp + pid, 0.0 * active)
        tl.store(tmp + num_rows + pid, 0.0 * active)
        tl.debug_barrier()

        # Compute preconditioned residual and rho
        local_rho = 0.0
        for row in range(block_start, block_end):
            # Load residual and compute preconditioned value
            start = tl.load(A_row_offsets + row)
            diag = tl.load(A_values + start)
            r_val = tl.load(r + row)
            z_val = r_val / (diag + (diag == 0.0))  # Avoid division by zero
            tl.store(z + row, z_val * active)

            # Accumulate rho locally
            local_rho += r_val * z_val * active

        tl.atomic_add(tmp, local_rho)
        tl.debug_barrier()

        rho = tl.load(tmp)

        # Update search direction using masked operations
        beta = rho / (rho_prev + (rho_prev == 0.0))
        beta = tl.where(iteration == 0, 0.0, beta)

        for row in range(block_start, block_end):
            z_val = tl.load(z + row)
            p_val = tl.load(p + row)
            new_p = z_val + beta * p_val
            p_val = tl.where(iteration == 0, z_val, new_p)
            tl.store(p + row, p_val * active)

        tl.debug_barrier()

        # Compute A*p and denominator
        local_denom = 0.0
        for row in range(block_start, block_end):
            ap_val = 0.0
            start = tl.load(A_row_offsets + row)
            end = tl.load(A_row_offsets + row + 1)

            for j in range(start, end):
                col = tl.load(A_column_indices + j)
                a_val = tl.load(A_values + j)
                p_val = tl.load(p + col)
                ap_val += a_val * p_val

            tl.store(tmp + row, ap_val * active)

            # Compute denominator contribution
            p_val = tl.load(p + row)
            local_denom += p_val * ap_val * active

        tl.atomic_add(tmp + num_rows, local_denom)
        tl.debug_barrier()

        # Compute alpha with safe division
        denom = tl.load(tmp + num_rows)
        alpha = rho / (denom + (denom == 0.0))
        alpha = alpha * active

        # Update solution and residual
        local_norm = 0.0
        for row in range(block_start, block_end):
            p_val = tl.load(p + row)
            ap_val = tl.load(tmp + row)

            # Update x
            x_val = tl.load(x + row)
            tl.store(x + row, x_val + alpha * p_val * active)

            # Update r and compute norm
            r_val = tl.load(r + row)
            r_new = r_val - alpha * ap_val
            tl.store(r + row, r_new * active)
            local_norm += r_new * r_new * active

        tl.atomic_add(tmp + num_rows + 1, local_norm)
        tl.debug_barrier()

        residual_norm = tl.sqrt(tl.load(tmp + num_rows + 1))
        converged = residual_norm < tolerance
        rho_prev = rho


# Example wrapper function to launch the kernel
def solve_pcg(
    A: csr_matrix,
    b: np.array,
    x_init: Optional[np.array] = None,
    max_iterations: int =1000,
    tolerance=1e-6,
):
    """
    Solve Ax = b using Preconditioned Conjugate Gradient method

    Parameters:
    - A: Sparse Matrix scipy.sparse.csr_matrix format
    - b: Right-hand side vector
    - x_init: Initial guess (optional)
    - max_iterations: Maximum CG iterations
    - tolerance: Convergence tolerance

    Returns:
    - Solved vector x
    """
    # Convert inputs to torch tensors if not already
    A_values = torch.as_tensor(A.data, dtype=torch.float32, device='cuda')
    A_row_offsets = torch.as_tensor(A.indptr, dtype=torch.int32, device='cuda')
    A_column_indices = torch.as_tensor(A.indices, dtype=torch.int32, device='cuda')
    b = torch.as_tensor(b, dtype=torch.float32, device='cuda')

    # Initialize solution vector
    if x_init is None:
        x = torch.zeros_like(b)
    else:
        x = torch.as_tensor(x_init, dtype=torch.float32, device='cuda')

    # Allocate working vectors
    r = b - sparse_matrix_vector_multiply(A_values, A_row_offsets, A_column_indices, x)
    p = r.clone()
    z = torch.zeros_like(r)
    tmp = torch.zeros(
        1, dtype=torch.float32, device="cuda"
    )  # Temporary reduction array

    # Configure kernel launch parameters
    num_rows = len(b)
    grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_SIZE"]),)

    # Launch Triton kernel
    pcg_kernel[grid](
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
        max_iterations,
        tolerance,
    )
    residual_norm = tmp
    return x, residual_norm


# Utility function for sparse matrix-vector multiplication
def sparse_matrix_vector_multiply(A_values, A_row_offsets, A_column_indices, x):
    """
    Perform sparse matrix-vector multiplication

    Parameters:
    - A_values: Non-zero values of sparse matrix
    - A_row_offsets: Row offsets in CSR format
    - A_column_indices: Column indices in CSR format
    - x: Input vector

    Returns:
    - Result of A * x
    """
    result = torch.zeros_like(x, device="cuda")
    for row in range(len(x)):
        row_start = A_row_offsets[row]
        row_end = A_row_offsets[row + 1]
        result[row] = torch.dot(
            A_values[row_start:row_end], x[A_column_indices[row_start:row_end]]
        )
    return result
