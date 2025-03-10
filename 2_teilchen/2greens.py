import numpy as np
import matplotlib.pyplot as plt

def greens_function_solution(n, sources, L=1, num_terms=50):
    """
    Compute the analytical solution to Poisson's equation using Green's function expansion
    for multiple point sources.
    
    Parameters:
    - n: Grid size (resolution)
    - sources: List of (x, y) positions of sources
    - L: Domain length (default: 1)
    - num_terms: Number of Fourier series terms (default: 50)
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
            
            source_sum = 0
            for (x_src, y_src) in sources:
                sin_mx0 = np.sin(m * np.pi * x_src / L)
                sin_ny0 = np.sin(n * np.pi * y_src / L)
                source_sum += sin_mx0 * sin_ny0
            
            u += (sin_mx * sin_ny * source_sum) / denom
    
    u *= -4 / L**2
    
    return X, Y, u

# Parameters
grid_size = 100
sources = [(0.3, 0.3), (0.7, 0.7)]  # Two sources at different locations

# Compute Analytical Solution
X, Y, U_analytical = greens_function_solution(grid_size, sources)

# Plot Analytical Solution
plt.imshow(U_analytical, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Solution u(x, y)')
plt.title("Analytical Solution using Green's Function (Two Sources)")
plt.show()

# Save Analytical Solution
np.save("analytical_solution_2sources.npy", U_analytical)
