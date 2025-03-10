import numpy as np
import matplotlib.pyplot as plt

def greens_function_solution(n, x_source, y_source, L=1, num_terms=50):
    """
    Compute the analytical solution to Poisson's equation using Green's function expansion.
    """
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    
    u = np.zeros_like(X)
    
    for m in range(1, num_terms + 1):
        for n in range(1, num_terms + 1):
            denom = (m**2 + n**2) * np.pi**2 / L**2
            
            sin_mx = np.sin(m * np.pi * X / L)
            sin_ny = np.sin(n * np.pi * Y / L)
            sin_mx0 = np.sin(m * np.pi * x_source / L)
            sin_ny0 = np.sin(n * np.pi * y_source / L)
            
            u += (sin_mx * sin_ny * sin_mx0 * sin_ny0) / denom
    
    u *= -4 / L**2
    
    return X, Y, u

# Parameters
grid_size = 100
x_src, y_src = 0.5, 0.5  # Source at the center of the domain

# Compute Analytical Solution
X, Y, U_analytical = greens_function_solution(grid_size, x_src, y_src)

# Plot Analytical Solution
plt.imshow(U_analytical, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Solution u(x, y)')
plt.title("Analytical Solution using Green's Function")
plt.show()



# Save Analytical solution
np.save("analytical_solution.npy", U_analytical)
