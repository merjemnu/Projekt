import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def laplacian_2d(n, h):
    """Creates the finite difference Laplacian matrix for a 2D grid with Dirichlet BCs."""
    N = n * n
    diag = -4 * np.ones(N)
    off_diag = np.ones(N)
    
    # Create sparse matrix for 2D Laplacian
    A = sp.diags([diag, off_diag, off_diag, off_diag, off_diag], [0, -1, 1, -n, n], shape=(N, N)).tolil()
    
    # Apply Dirichlet boundary conditions (G = 0 at boundary)
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:  # Boundary points
                A[index, :] = 0  # Zero out the row
                A[index, index] = 1  # Set diagonal to 1 to enforce G = 0

    return A.tocsr() / (h * h)

def greens_function(n, x_source, y_source):
    h = 1.0 / (n - 1)
    A = laplacian_2d(n, h)
    
    # Delta function at (x_source, y_source)
    b = np.zeros(n * n)
    source_index = x_source * n + y_source
    b[source_index] = 1.0 / (h * h)
    
    # Solve linear system
    G = spla.spsolve(A, b)
    return G.reshape((n, n))

# Parameters
grid_size = 200
x_src, y_src = 10, 20  # Source at center

# Compute Green's function
G = greens_function(grid_size, x_src, y_src)

# Plot
plt.imshow(G, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
plt.colorbar(label="Green's Function G(x, x')")
plt.title("Numerical Green's Function for Laplace Equation")
plt.show()