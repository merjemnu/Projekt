import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def fem_laplacian(n):
    """Create finite element Laplacian matrix with Dirichlet BCs."""
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
    source_index = int(y_source * n + x_source)  # Point source location
    b[source_index] = 1.0 / (h * h)  # Delta function approximation
    
    u = spla.spsolve(A, b)
    return u.reshape((n, n))

def analytical_solution(n):
    """Compute analytical Green's function solution for Dirichlet BCs."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    X0, Y0 = 0.5, 0.5  # Point source at center

    # Green's function with Dirichlet BCs (approximated)
    G = -1 / (2 * np.pi) * np.log(np.sqrt((X - X0)**2 + (Y - Y0)**2))
    G -= np.min(G)  # Shift to satisfy Dirichlet BCs (zero at boundaries)
    
    return G

# Grid sizes to test
grid_sizes = [20, 40, 80, 160]
errors = []

for N in grid_sizes:
    U_fem = fem_poisson_solver(N, N//2, N//2)  # FEM solution
    U_analytical = analytical_solution(N)      # Analytical solution

    # Compute absolute error and L2 norm
    error = np.abs(U_fem - U_analytical)
    error_norm = np.linalg.norm(error) / np.sqrt(N**2)  # L2 error per grid point
    errors.append(error_norm)

# Plot Error Convergence
plt.figure(figsize=(7, 5))
plt.loglog(grid_sizes, errors, 'o-', label="L2 Norm of Error")
plt.xlabel("Grid Size (N)")
plt.ylabel("Error Norm")
plt.title("Error Convergence of FEM Solution")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.show()
