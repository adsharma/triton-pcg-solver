import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from scipy.sparse import csr_matrix


@triton.jit
def pcg_init_kernel(
    A_values,
    A_row_offsets,
    A_column_indices,
    b,
    x,
    r,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Initialize residual for PCG method"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, num_rows)

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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["num_rows"],
)
@triton.jit
def pcg_iteration_kernel(
    A_values,
    A_row_offsets,
    A_column_indices,
    x,
    r,
    p,
    z,
    tmp,
    num_rows,
    beta,
    first_iteration: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single iteration of Preconditioned Conjugate Gradient method"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, num_rows)

    # Reset reduction variable
    tl.store(tmp, 0.0)
    tl.debug_barrier()

    # Compute preconditioned residual and rho
    local_rho = 0.0
    for row in range(block_start, block_end):
        start = tl.load(A_row_offsets + row)
        diag = tl.load(A_values + start)
        r_val = tl.load(r + row)
        z_val = r_val / (diag + (diag == 0.0))  # Avoid division by zero
        tl.store(z + row, z_val)
        local_rho += r_val * z_val

        # Update search direction p
        if first_iteration:
            tl.store(p + row, z_val)
        else:
            p_val = tl.load(p + row)
            tl.store(p + row, z_val + beta * p_val)

    tl.atomic_add(tmp, local_rho)
    tl.debug_barrier()

    rho = tl.load(tmp)
    tl.store(tmp, 0.0)
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

        tl.store(z + row, ap_val)  # Reuse z as temporary storage
        p_val = tl.load(p + row)
        local_denom += p_val * ap_val

    tl.atomic_add(tmp, local_denom)
    tl.debug_barrier()

    # Update solution and residual
    denom = tl.load(tmp)
    alpha = rho / (denom + 1e-15)
    local_norm = 0.0

    for row in range(block_start, block_end):
        p_val = tl.load(p + row)
        ap_val = tl.load(z + row)  # Read back from temporary storage

        # Update x
        x_val = tl.load(x + row)
        tl.store(x + row, x_val + alpha * p_val)

        # Update r and compute norm
        r_val = tl.load(r + row)
        r_new = r_val - alpha * ap_val
        tl.store(r + row, r_new)
        local_norm += r_new * r_new

    tl.atomic_add(tmp, local_norm)


def solve_pcg(
    A: csr_matrix,
    b: np.array,
    x_init: Optional[np.array] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[torch.Tensor, float]:
    """
    Solve Ax = b using Preconditioned Conjugate Gradient method with Python-based iteration control

    Parameters:
    - A: Sparse Matrix in scipy.sparse.csr_matrix format
    - b: Right-hand side vector
    - x_init: Initial guess (optional)
    - max_iterations: Maximum CG iterations
    - tolerance: Convergence tolerance

    Returns:
    - Tuple(x, sigma) where:
        x: Solution vector
        sigma: Final residual norm
    """
    # Convert inputs to torch tensors
    A_values = torch.as_tensor(A.data, dtype=torch.float32, device="cuda")
    A_row_offsets = torch.as_tensor(A.indptr, dtype=torch.int32, device="cuda")
    A_column_indices = torch.as_tensor(A.indices, dtype=torch.int32, device="cuda")
    b = torch.as_tensor(b, dtype=torch.float32, device="cuda")

    # Initialize solution vector
    if x_init is None:
        x = torch.zeros_like(b)
    else:
        x = torch.as_tensor(x_init, dtype=torch.float32, device="cuda")

    # Allocate working vectors
    num_rows = len(b)
    r = torch.zeros_like(b)
    p = r.clone()
    z = torch.zeros_like(r)
    tmp = torch.zeros(1, dtype=torch.float32, device="cuda")

    # Configure kernel grid
    grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_SIZE"]),)

    # Initialize residual
    pcg_init_kernel[grid](
        A_values, A_row_offsets, A_column_indices, b, x, r, num_rows, BLOCK_SIZE=256
    )

    # Main iteration loop
    rho = rho_prev = 0.0
    for iteration in range(max_iterations):
        pcg_iteration_kernel[grid](
            A_values,
            A_row_offsets,
            A_column_indices,
            x,
            r,
            p,
            z,
            tmp,
            num_rows,
            beta=0.0 if iteration == 0 else (rho / (rho_prev + 1e-15)),
            first_iteration=(iteration == 0),
        )

        # Get values for next iteration
        rho = tmp[0].item()  # Store rho for beta calculation
        rho_prev = tmp[0].item()  # Store denominator for beta calculation

        # Calculate residual norm
        sigma = math.sqrt(tmp[0].item())
        if sigma < tolerance:
            break

    return x, sigma
