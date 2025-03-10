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
            index = i * n + j  #converting a 2d grid size n * x with row major ordering index = rox * n = col
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[index, :] = 0
                A[index, index] = 1
    return A.tocsc()

def fem_poisson_solver(n, x_source, y_source):
    """Solve Poisson equation using FEM."""
    h = 1.0 / (n + 1)
    A = fem_laplacian(n) / (h * h)
    
    b = np.zeros(n * n)
    source_index = int(y_source * n + x_source)  # (x, y) corresponds to (col, row)
    b[source_index] = 1.0 / (h * h)  # like the delta function 
    
    u = spla.spsolve(A, b)
    return u.reshape((n, n))

# Parameters
grid_size = 100
x_src, y_src = grid_size // 2, grid_size // 2  # Centeringggg the source

# Compute FEM-based Poisson solution
U = fem_poisson_solver(grid_size, x_src, y_src)

# Plot
plt.imshow(U, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Solution u(x, y)')
plt.title("FEM Solution for Poisson Equation")
plt.show()

np.save("fem_solution.npy", U)
