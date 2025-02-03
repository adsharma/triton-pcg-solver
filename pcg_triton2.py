import triton
import triton.language as tl
import torch

@triton.jit
def pcg_kernel(
    A_values, A_row_offsets, A_column_indices,  # Sparse matrix representation
    b,                                          # Right-hand side vector
    x,                                          # Solution vector (input/output)
    r,                                          # Residual vector
    p,                                          # Search direction vector
    z,                                          # Preconditioned residual vector
    tmp,                                        # Temporary working array
    num_rows,                                   # Number of rows in the matrix
    max_iterations,                             # Maximum number of iterations
    tolerance,                                  # Convergence tolerance
    BLOCK_SIZE: tl.constexpr                    # Compile-time block size
):
    """
    Preconditioned Conjugate Gradient kernel for solving linear systems Ax = b
    
    Parameters:
    - Sparse matrix A in Compressed Sparse Row (CSR) format
    - b: right-hand side vector
    - x: initial guess / solution vector (modified in-place)
    - r: residual vector
    - p: search direction vector
    - z: preconditioned residual vector
    - tmp: temporary working array for reductions
    - num_rows: number of rows in the matrix
    """
    # Global thread and block information
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, num_rows)

    # Iteration-specific variables
    alpha = 0.0  # Step size
    beta = 0.0   # Conjugate gradient beta parameter
    rho = 0.0    # Current residual norm squared
    rho_prev = 0.0  # Previous residual norm squared

    converged = False

    # Main PCG iteration loop
    for iteration in range(max_iterations):
        residual_norm = 0.0
        if not converged:
            # Compute preconditioned residual
            for row in range(block_start, block_end):
                # Precondition
                # Assumes diagonal is stored first in each row
                off = tl.load(A_row_offsets + row)
                diag = tl.load(A_values + off)
                r_row = tl.load(r + row)
                val = r_row / diag if diag != 0 else r_row
                tl.store(z + row, val)
        
            # Compute inner product (rho)
            rho_prev = rho
            rho = 0.0
            for row in range(block_start, block_end):
                r_row = tl.load(r + row)
                z_row = tl.load(z + row)
                rho += r_row * z_row
        
            # Synchronize and reduce rho across blocks
            tl.atomic_add(tmp, rho)
        
            # Update search direction on first iteration or restart
            if iteration == 0 or rho == 0:
                for row in range(block_start, block_end):
                    z_row = tl.load(z + row)
                    tl.store(p+row, z_row)
            else:
                # Compute beta for conjugate gradient restart
                beta = rho / rho_prev
                for row in range(block_start, block_end):
                    z_row = tl.load(z + row)
                    p_row = tl.load(p + row)
                    tl.store(p+row, z_row + beta * p_row)
        
            # Compute A * p
            for row in range(block_start, block_end):
                tl.store(tmp + row, 0.0)
                start = tl.load(A_row_offsets + row)
                end = tl.load(A_row_offsets + row + 1)
                for j in range(start, end):
                    a_val = tl.load(A_values + j)
                    p_val = tl.load(p + j)
                    val = a_val * p_val
                    tl.atomic_add(tmp + row, val)
        
            # Compute step size (alpha)
            denominator = 0.0
            for row in range(block_start, block_end):
                p_row = tl.load(p + row)
                t_row = tl.load(tmp + row)
                denominator += (p_row * t_row)
        
            # Atomic reduction for denominator
            tl.atomic_add(tmp, denominator)
        
            if denominator != 0:
                alpha = rho / denominator
        
            # Update solution and residual
            for row in range(block_start, block_end):
                p_row = tl.load(p + row)
                t_row = tl.load(tmp + row)
                tl.atomic_add(x + row, alpha * p_row)
                tl.atomic_add(r + row, -alpha * t_row)
        
            # Check convergence
            for row in range(block_start, block_end):
                r_row = tl.load(r + row)
                residual_norm += r_row * r_row
        
            # Atomic reduction for residual norm
            tl.atomic_add(tmp, -residual_norm)
        
        # Break if converged
        converged = tl.sqrt(residual_norm) < tolerance

# Example wrapper function to launch the kernel
def solve_pcg(A_values, A_row_offsets, A_column_indices, b, 
               x_init=None, max_iterations=1000, tolerance=1e-6):
    """
    Solve Ax = b using Preconditioned Conjugate Gradient method
    
    Parameters:
    - A_values: Non-zero values of sparse matrix
    - A_row_offsets: Row offsets in CSR format
    - A_column_indices: Column indices in CSR format
    - b: Right-hand side vector
    - x_init: Initial guess (optional)
    - max_iterations: Maximum CG iterations
    - tolerance: Convergence tolerance
    
    Returns:
    - Solved vector x
    """
    # Convert inputs to torch tensors if not already
    A_values = torch.as_tensor(A_values, dtype=torch.float32)
    A_row_offsets = torch.as_tensor(A_row_offsets, dtype=torch.int32)
    A_column_indices = torch.as_tensor(A_column_indices, dtype=torch.int32)
    b = torch.as_tensor(b, dtype=torch.float32)
    
    # Initialize solution vector
    if x_init is None:
        x = torch.zeros_like(b)
    else:
        x = torch.as_tensor(x_init, dtype=torch.float32)
    
    # Allocate working vectors
    r = b - sparse_matrix_vector_multiply(A_values, A_row_offsets, A_column_indices, x)
    p = r.clone()
    z = torch.zeros_like(r)
    tmp = torch.zeros(1, dtype=torch.float32, device='cuda')  # Temporary reduction array
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    grid = ((-(-len(b) // BLOCK_SIZE)),)
    
    # Launch Triton kernel
    pcg_kernel[grid](
        A_values, A_row_offsets, A_column_indices,
        b, x, r, p, z, tmp,
        len(b),
        max_iterations,
        tolerance,
        BLOCK_SIZE
    )
    
    return x

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
    result = torch.zeros_like(x, device='cuda')
    for row in range(len(x)):
        row_start = A_row_offsets[row]
        row_end = A_row_offsets[row + 1]
        result[row] = torch.dot(
            A_values[row_start:row_end], 
            x[A_column_indices[row_start:row_end]]
        )
    return result
