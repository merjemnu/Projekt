import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
#x, y = np.meshgrid(x, y)

# Define sources and strengths
sources = [(0, 2)]  # Example source at the origin
strengths = [1]     # Example strength

# Green's function for 2D Poisson
def greens_function_2d(x, y, sources, strengths):
    potential = np.zeros_like(x)
    for (sx, sy), strength in zip(sources, strengths):
        r = np.sqrt((x - sx)**2 + (y - sy)**2)
        potential += strength / (2 * np.pi * np.maximum(r, 1e-12))
    return potential

# Evaluate the Green's function on the grid
potential = greens_function_2d(x, y, sources, strengths)

# Plot the potential
plt.contourf(x, y, potential, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Green's Function Potential")
plt.show()