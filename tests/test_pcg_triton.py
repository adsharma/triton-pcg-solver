import numpy as np
import pytest
import scipy.sparse as sp
import torch

from pcg.solver import solve_pcg


@pytest.fixture
def sparse_csr_matrix():
    return sp.csr_matrix(
        np.array(
            [
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 2],
            ]
        )
    )


# Define the test function
def test_pcg_kernel(sparse_csr_matrix):
    # Define the matrix dimensions
    num_rows, num_cols = sparse_csr_matrix.shape

    # Create random vectors
    # b_orig = b = np.random.rand(num_rows).astype(np.float32)
    b_orig = b = np.array([1, 0, 0, 1])
    x = np.random.rand(num_cols).astype(np.float32)
    r = np.random.rand(num_rows).astype(np.float32)
    p = np.random.rand(num_rows).astype(np.float32)
    z = np.random.rand(num_rows).astype(np.float32)
    tmp = np.random.rand(num_rows).astype(np.float32)

    # Move the data to the GPU
    b = torch.tensor(b, dtype=torch.float32, device="cuda")
    x = torch.tensor(x, dtype=torch.float32, device="cuda")
    r = torch.tensor(r, dtype=torch.float32, device="cuda")
    p = torch.tensor(p, dtype=torch.float32, device="cuda")
    z = torch.tensor(z, dtype=torch.float32, device="cuda")
    tmp = torch.tensor(tmp, dtype=torch.float32, device="cuda")

    # Launch the PCG kernel
    x, sigma = solve_pcg(sparse_csr_matrix, b, x)

    # Synchronize the GPU
    torch.cuda.synchronize()

    x_cpu = x.to("cpu")
    x_expected = np.array([0.5, 0, 0, 0.5])
    assert np.allclose(x_expected, x_cpu)
    A_dense = torch.tensor(sparse_csr_matrix.todense(), dtype=torch.float32)
    b_actual = A_dense @ x_cpu
    assert np.allclose(b_actual.numpy(), b_orig)
