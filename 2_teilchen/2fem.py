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
            index = i * n + j  # Converting a 2D grid size n * x with row-major ordering
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[index, :] = 0
                A[index, index] = 1
    return A.tocsc()

def fem_poisson_solver(n, sources):
    """Solve Poisson equation using FEM for multiple sources."""
    h = 1.0 / (n + 1)
    A = fem_laplacian(n) / (h * h)
    
    b = np.zeros(n * n)
    for (x_source, y_source, charge) in sources:
        source_index = int(y_source * n + x_source)  # (x, y) corresponds to (col, row)
        b[source_index] = charge / (h * h)  # Like the delta function 
    
    u = spla.spsolve(A, b)
    return u.reshape((n, n))

# Parameters
grid_size = 100
sources = [
    (grid_size // 3, grid_size // 2, 1.0),  # First charge (positive)
    (2 * grid_size // 3, grid_size // 2, -1.0)  # Second charge (negative)
]

# Compute FEM-based Poisson solution
U = fem_poisson_solver(grid_size, sources)

# Plot
plt.imshow(U, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Solution u(x, y)')
plt.title("FEM Solution for Poisson Equation with Two Charges")
plt.show()

np.save("fem_solution_two_charges.npy", U)
