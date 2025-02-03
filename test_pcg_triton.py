import triton
import numpy as np
import torch
from pcg_triton import launch_pcg_kernel

# Define the test function
def test_pcg_kernel():
    # Define the matrix dimensions
    num_rows = 1024
    num_cols = 1024

    # Create a random sparse matrix
    A_values = np.random.rand(num_rows * 10).astype(np.float32)
    A_row_offsets = np.random.randint(0, 10, size=num_rows + 1).astype(np.int32)
    A_column_indices = np.random.randint(0, num_cols, size=num_rows * 10).astype(np.int32)

    # Create random vectors
    b = np.random.rand(num_rows).astype(np.float32)
    x = np.random.rand(num_cols).astype(np.float32)
    r = np.random.rand(num_rows).astype(np.float32)
    p = np.random.rand(num_rows).astype(np.float32)
    z = np.random.rand(num_rows).astype(np.float32)
    tmp = np.random.rand(num_rows).astype(np.float32)

    # Move the data to the GPU
    A_values = torch.tensor(A_values, dtype=torch.float32, device='cuda')
    A_row_offsets = torch.tensor(A_row_offsets, dtype=torch.int32, device='cuda')
    A_column_indices = torch.tensor(A_column_indices, dtype=torch.int32, device='cuda')
    b = torch.tensor(b, dtype=torch.float32, device='cuda')
    x = torch.tensor(x, dtype=torch.float32, device='cuda')
    r = torch.tensor(r, dtype=torch.float32, device='cuda')
    p = torch.tensor(p, dtype=torch.float32, device='cuda')
    z = torch.tensor(z, dtype=torch.float32, device='cuda')
    tmp = torch.tensor(tmp, dtype=torch.float32, device='cuda')

    # Launch the PCG kernel
    launch_pcg_kernel(A_values, A_row_offsets, A_column_indices, b, x, r, p, z, tmp, num_rows)

    # Synchronize the GPU
    triton.synchronize()

    # Check the results
    r_host = r.to('cpu').numpy()
    p_host = p.to('cpu').numpy()
    z_host = z.to('cpu').numpy()
    assert np.allclose(r_host, np.random.rand(num_rows).astype(np.float32))
    assert np.allclose(p_host, np.random.rand(num_rows).astype(np.float32))
    assert np.allclose(z_host, np.random.rand(num_rows).astype(np.float32))

# Run the test
test_pcg_kernel()
