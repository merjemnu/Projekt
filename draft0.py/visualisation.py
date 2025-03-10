import numpy as np
import matplotlib.pyplot as plt

# Green's function for 2D Poisson
def greens_function_2d(grid_x, grid_y, sources, strengths):
    potential = np.zeros_like(grid_x)
    for (sx, sy), strength in zip(sources, strengths):
        r = np.sqrt((grid_x - sx)**2 + (grid_y - sy)**2)
        potential += strength / (2 * np.pi * np.maximum(r, 1e-12))
    return potential




# Compute Electric Field
def compute_electric_field(grid_x, grid_y, potential):
    grad_x, grad_y = np.gradient(potential, grid_x[0, 1] - grid_x[0, 0], grid_y[1, 0] - grid_y[0, 0])
    return -grad_x, -grad_y

# Biot-Savart Law for Magnetic Field
def biot_savart_2d(grid_x, grid_y, sources, currents):
    B_z = np.zeros_like(grid_x)
    mu_0 = 4 * np.pi * 1e-7
    for (sx, sy), current in zip(sources, currents):
        r_x = grid_x - sx
        r_y = grid_y - sy
        r_squared = r_x**2 + r_y**2
        cross_product_z = r_x * (-current) - r_y * 0
        B_z += mu_0 / (2 * np.pi) * cross_product_z / np.maximum(r_squared, 1e-12)
    return B_z

# Define the grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
grid_x, grid_y = np.meshgrid(x, y)

# Define sources
sources = [(0, 0), (2, 2)]
strengths = [1, -1]  # Charge strengths
currents = [1, -1]   # Current magnitudes

# Compute potential and fields
potential = greens_function_2d(grid_x, grid_y, sources, strengths)
E_x, E_y = compute_electric_field(grid_x, grid_y, potential)
B_z = biot_savart_2d(grid_x, grid_y, sources, currents)

# Pl ot results
plt.figure(figsize=(16, 5))

# Plot Electric Potential
plt.subplot(1, 3, 1)
plt.contourf(grid_x, grid_y, potential, levels=50, cmap='RdBu_r')
plt.colorbar(label='Potential (φ)')
plt.title("Electric Potential")
plt.scatter(*zip(*sources), color='black', zorder=10)

# Plot Electric Field
plt.subplot(1, 3, 2)
plt.quiver(grid_x, grid_y, E_x, E_y, scale=50, color='blue')
plt.title("Electric Field (E)")

# Plot Magnetic Field
plt.subplot(1, 3, 3)
plt.contourf(grid_x, grid_y, B_z, levels=50, cmap='viridis')
plt.colorbar(label='Magnetic Field (B_z)')
plt.title("Magnetic Field (B)")

plt.tight_layout()
plt.show()