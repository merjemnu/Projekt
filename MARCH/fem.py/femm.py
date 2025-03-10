import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def generate_mesh(n):
    """Generate FEM mesh grid."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    return X, Y

def fem_laplacian(n):
    """Create finite element Laplacian matrix."""
    N = n * n
    diag = -4 * np.ones(N)
    off_diag = np.ones(N)
    A = sp.diags([diag, off_diag, off_diag, off_diag, off_diag], [0, -1, 1, -n, n], shape=(N, N)).tolil()
    
    # Apply Dirichlet boundary conditions
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[index, :] = 0
                A[index, index] = 1
    return A.tocsc()

def fem_poisson_solver(n, x_source, y_source):
    """Solve Poisson equation using FEM."""
    h = 1.0 / (n + 1)
    A = fem_laplacian(n) / (h * h)
    
    b = np.zeros(n * n)

    # Distribute source over nearby points to avoid artifacts
    src_idx = x_source * n + y_source
    b[max(0, src_idx - 1)] += 0.25 / (h * h)
    b[src_idx] += 0.5 / (h * h)
    b[min(n * n - 1, src_idx + 1)] += 0.25 / (h * h)

    # Solve using Conjugate Gradient for large matrices
    #u, _ = spla.cg(A, b, tol=1e-6)
    u, _ = spla.cg(A, b, rtol=1e-6)

    return u.reshape((n, n))

# Parameters
grid_size = 100  # Increased for better resolution
x_src, y_src = grid_size // 2, grid_size // 2  # Place source in center

# Compute FEM-based Poisson solution
U = fem_poisson_solver(grid_size, x_src, y_src)

# Plot
plt.imshow(U, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', interpolation='bicubic')  # Smooth interpolation
plt.colorbar(label='Solution u(x, y)')
plt.title("Optimized FEM Solution for Poisson Equation")
plt.show()
